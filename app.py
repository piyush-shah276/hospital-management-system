import os, io, json, sqlite3, datetime as dt
from functools import wraps
from flask import send_from_directory, Flask, render_template, request, redirect, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import joblib, pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from ml.train import FEATS, train_all

B = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(B, 'database', 'hospital.db')
app = Flask(__name__); app.secret_key = 'hmdps-change-this-secret'
HOSP_GSTIN = '05AAAAA0000A1Z5'   # <- apna asli GSTIN yahan daalo (ye sirf sample hai)
MED_GST = 12   # medicines par GST % (bill amount GST-inclusive maana gaya hai)
SERVICE_GST = 0   # health services (consultation/tests) GST-exempt
HOSP = 'S.P. Multispeciality Hospital'   # <- hospital ka naam yahin se badlo (UI + PDF dono mein)

@app.context_processor
def inject(): return {'hospital': HOSP}
NOW = lambda: dt.datetime.now().strftime('%Y-%m-%d %H:%M')

def q(sql, a=(), one=False, commit=False):
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; r = c.execute(sql, a)
    if commit: c.commit(); i = r.lastrowid; c.close(); return i
    rows = r.fetchall(); c.close(); return (rows[0] if rows else None) if one else rows

def auth(*roles, pt=False):
    def deco(f):
        @wraps(f)
        def w(*a, **k):
            if 'user' not in session: return redirect('/login')
            if session['role'] == 'patient' and not pt: return redirect(f"/patient/{session['pid']}")
            if roles and session['role'] not in roles and not (pt and session['role'] == 'patient'): flash('Access denied for your role.'); return redirect('/')
            return f(*a, **k)
        return w
    return deco

def migrate():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS staff(id INTEGER PRIMARY KEY, name TEXT, post TEXT, department TEXT, salary REAL, phone TEXT, joined TEXT, status TEXT DEFAULT 'Active')")
    for col in ('active INT DEFAULT 1', 'patient_id INT'):
        try: c.execute('ALTER TABLE users ADD COLUMN ' + col)
        except sqlite3.OperationalError: pass
    if c.execute('SELECT COUNT(*) FROM staff').fetchone()[0] == 0:
        c.executemany('INSERT INTO staff(name,post,department,salary,phone,joined) VALUES(?,?,?,?,?,?)', [
            ('Dr. Neha Sharma','Doctor','General Medicine',90000,'9876500001','2024-04-01'),('Dr. Rahul Verma','Doctor','Cardiology',120000,'9876500002','2023-08-15'),('Dr. Anil Gupta','Doctor','Endocrinology',110000,'9876500003','2023-01-10'),
            ('Sunita Rawat','Nurse','General Ward',32000,'9811100001','2024-06-01'),('Meena Joshi','Nurse','ICU',36000,'9811100002','2023-11-20'),('Pooja Negi','Nurse','OPD',30000,'9811100003','2025-02-01'),('Kavita Bisht','Nurse','Cardiology',34000,'9811100004','2024-09-12'),
            ('Riya Thakur','Receptionist','Front Desk',22000,'9811100005','2025-01-05'),('Mohit Saini','Pharmacist','Pharmacy',28000,'9811100006','2024-03-18')])
    seed_meds(c); c.commit(); c.close()

MEDS = [('Metformin 500mg','diabetes',35,'#0f766e'),('Glimepiride 1mg','diabetes',60,'#2563eb'),('Sitagliptin 50mg','diabetes',180,'#7c3aed'),('Aspirin 75mg','heart',20,'#dc2626'),('Atorvastatin 10mg','heart',95,'#ea580c'),('Metoprolol 25mg','heart',55,'#be185d'),
        ('Amlodipine 5mg','hypertension',30,'#0891b2'),('Telmisartan 40mg','hypertension',70,'#4d7c0f'),('Silymarin 140mg','liver',120,'#b45309'),('Ursodeoxycholic Acid 300mg','liver',210,'#475569')]

def pill(path, name, color):
    open(path, 'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 140"><rect width="200" height="140" rx="12" fill="#f0fdfa"/><g transform="rotate(-30 100 60)"><rect x="55" y="40" width="90" height="40" rx="20" fill="{color}"/><rect x="100" y="40" width="45" height="40" rx="20" fill="#fff" stroke="{color}" stroke-width="3"/></g><text x="100" y="124" font-family="Arial" font-size="12" text-anchor="middle" fill="#0f172a">{name}</text></svg>')

def seed_meds(c):
    import re
    d = os.path.join(B, 'static', 'medicines'); os.makedirs(d, exist_ok=True); pill(os.path.join(d, 'default.svg'), 'Medicine', '#64748b')
    c.execute('CREATE TABLE IF NOT EXISTS medicines(id INTEGER PRIMARY KEY, name TEXT, disease TEXT, price REAL, image TEXT)')
    if c.execute('SELECT COUNT(*) FROM medicines').fetchone()[0] == 0:
        for n, dz, p, col in MEDS:
            f = re.sub(r'[^a-z0-9]+', '_', n.lower()) + '.svg'; pill(os.path.join(d, f), n, col)
            c.execute('INSERT INTO medicines(name,disease,price,image) VALUES(?,?,?,?)', (n, dz, p, f))

def mine(aid):
    if session['role'] != 'patient': return True
    r = q('SELECT patient_id FROM appointments WHERE id=?', (aid,), one=True); return bool(r and r['patient_id'] == session['pid'])

def init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if not all(os.path.exists(os.path.join(B, 'ml', 'models', d + '.joblib')) for d in FEATS): train_all()
    if os.path.exists(DB): migrate(); return
    c = sqlite3.connect(DB); c.executescript('''
    CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT, name TEXT);
    CREATE TABLE doctors(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, phone TEXT);
    CREATE TABLE patients(id INTEGER PRIMARY KEY, name TEXT, age INT, gender TEXT, phone TEXT, blood TEXT, created TEXT);
    CREATE TABLE appointments(id INTEGER PRIMARY KEY, patient_id INT, doctor_id INT, date TEXT, time TEXT, reason TEXT, status TEXT DEFAULT 'Scheduled');
    CREATE TABLE predictions(id INTEGER PRIMARY KEY, patient_id INT, disease TEXT, label TEXT, risk REAL, vitals TEXT, created TEXT);
    CREATE TABLE bills(id INTEGER PRIMARY KEY, appointment_id INT, item TEXT, amount REAL);''')
    for u, p, r, n in [('admin','admin123','admin','Administrator'),('doctor1','doctor123','doctor','Dr. Neha Sharma'),('recep1','recep123','receptionist','Riya (Reception)')]:
        c.execute('INSERT INTO users(username,password,role,name) VALUES(?,?,?,?)', (u, generate_password_hash(p), r, n))
    c.executemany('INSERT INTO doctors(name,dept,phone) VALUES(?,?,?)', [('Dr. Neha Sharma','General Medicine','9876500001'),('Dr. Rahul Verma','Cardiology','9876500002'),('Dr. Anil Gupta','Endocrinology','9876500003')])
    t = dt.date.today().isoformat()
    c.executemany('INSERT INTO patients(name,age,gender,phone,blood,created) VALUES(?,?,?,?,?,?)', [('Ramesh Kumar',52,'Male','9810011111','B+',NOW()),('Sunita Devi',47,'Female','9810022222','O+',NOW()),('Aman Singh',29,'Male','9810033333','A+',NOW())])
    c.executemany('INSERT INTO appointments(patient_id,doctor_id,date,time,reason) VALUES(?,?,?,?,?)', [(1,2,t,'10:00','Chest discomfort'),(2,3,t,'11:30','Routine sugar check')])
    c.commit(); c.close(); migrate()

@app.route('/sw.js')
def sw(): return send_from_directory(app.static_folder, 'sw.js')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = q('SELECT * FROM users WHERE username=?', (request.form['username'],), one=True)
        if u and u['active'] != 0 and check_password_hash(u['password'], request.form['password']):
            session.update(user=u['username'], role=u['role'], name=u['name'], pid=u['patient_id']); return redirect('/')
        flash('Wrong username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect('/login')

@app.route('/')
@auth()
def dash():
    t = dt.date.today().isoformat(); n = lambda s, a=(): q(s, a, one=True)[0]
    stats = [('Patients', n('SELECT COUNT(*) FROM patients')), ('Doctors', n('SELECT COUNT(*) FROM doctors')), ("Today's Appointments", n('SELECT COUNT(*) FROM appointments WHERE date=?', (t,))),
             ('At-risk Flags', n("SELECT COUNT(*) FROM predictions WHERE label='At risk'")), ('Revenue (Rs)', round(n('SELECT COALESCE(SUM(amount),0) FROM bills')))]
    flagged = q("SELECT p.*, pt.name pname FROM predictions p JOIN patients pt ON pt.id=p.patient_id WHERE label='At risk' ORDER BY p.id DESC LIMIT 8")
    today = q('SELECT a.*, pt.name pname, d.name dname FROM appointments a JOIN patients pt ON pt.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id WHERE date=?', (t,))
    return render_template('dashboard.html', stats=stats, flagged=flagged, today=today, metrics=json.load(open(os.path.join(B, 'ml', 'models', 'metrics.json'))))

@app.route('/patients', methods=['GET','POST'])
@auth()
def patients():
    if request.method == 'POST' and session['role'] in ('admin','receptionist'):
        f = request.form; i = q('INSERT INTO patients(name,age,gender,phone,blood,created) VALUES(?,?,?,?,?,?)', (f['name'], f['age'], f['gender'], f['phone'], f['blood'], NOW()), commit=True)
        flash('Patient registered.'); return redirect(f'/patient/{i}')
    s = '%' + request.args.get('s', '') + '%'
    return render_template('patients.html', rows=q('SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ? ORDER BY id DESC', (s, s)))

@app.route('/patient/<int:pid>')
@auth(pt=True)
def patient(pid):
    if session['role'] == 'patient' and session['pid'] != pid: return redirect(f"/patient/{session['pid']}")
    p = q('SELECT * FROM patients WHERE id=?', (pid,), one=True)
    appts = q('SELECT a.*, d.name dname FROM appointments a JOIN doctors d ON d.id=a.doctor_id WHERE patient_id=? ORDER BY a.id DESC', (pid,))
    bills = {a['id']: q('SELECT * FROM bills WHERE appointment_id=?', (a['id'],)) for a in appts}
    preds = [dict(r, v=json.loads(r['vitals'])) for r in q('SELECT * FROM predictions WHERE patient_id=? ORDER BY id DESC', (pid,))]
    return render_template('patient.html', p=p, appts=appts, bills=bills, preds=preds, feats=FEATS, meds=q('SELECT * FROM medicines ORDER BY disease, name'))

@app.route('/predict/<int:pid>', methods=['POST'])
@auth('admin','doctor')
def predict(pid):
    d = request.form['disease']; cols = [c for c, _ in FEATS[d]]
    vals = [float(request.form[c]) for c in cols]
    risk = float(joblib.load(os.path.join(B, 'ml', 'models', d + '.joblib')).predict_proba(pd.DataFrame([vals], columns=cols))[0][1]) * 100
    q('INSERT INTO predictions(patient_id,disease,label,risk,vitals,created) VALUES(?,?,?,?,?,?)', (pid, d, 'At risk' if risk >= 50 else 'Not at risk', round(risk, 1), json.dumps(dict(zip(cols, vals))), NOW()), commit=True)
    flash('Prediction saved to patient record.'); return redirect(f'/patient/{pid}')

@app.route('/doctors', methods=['GET','POST'])
@auth()
def doctors():
    if request.method == 'POST' and session['role'] == 'admin':
        f = request.form; q('INSERT INTO doctors(name,dept,phone) VALUES(?,?,?)', (f['name'], f['dept'], f['phone']), commit=True); flash('Doctor added.')
    rows = q('SELECT * FROM doctors')
    return render_template('doctors.html', rows=rows, sigs={r['id'] for r in rows if os.path.exists(os.path.join(B, 'static', 'signatures', f"{r['id']}.png"))})

@app.route('/appointments', methods=['GET','POST'])
@auth()
def appointments():
    if request.method == 'POST' and session['role'] in ('admin','receptionist'):
        f = request.form; q('INSERT INTO appointments(patient_id,doctor_id,date,time,reason) VALUES(?,?,?,?,?)', (f['patient_id'], f['doctor_id'], f['date'], f['time'], f['reason']), commit=True); flash('Appointment scheduled.')
    rows = q('SELECT a.*, pt.name pname, d.name dname FROM appointments a JOIN patients pt ON pt.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id ORDER BY date DESC, time')
    return render_template('appointments.html', rows=rows, pts=q('SELECT id,name FROM patients'), docs=q('SELECT id,name,dept FROM doctors'), today=dt.date.today().isoformat())

@app.route('/appt/<int:aid>/done')
@auth('admin','doctor')
def done(aid): q("UPDATE appointments SET status='Completed' WHERE id=?", (aid,), commit=True); return redirect(request.referrer or '/appointments')


def hdr(c, title):
    c.setFillColorRGB(.06,.46,.43); c.rect(0,770,595,72,fill=1,stroke=0); c.setFillColorRGB(1,1,1); c.circle(55,806,24,fill=1,stroke=0)
    c.setFillColorRGB(.86,.15,.15); c.rect(49,794,12,24,fill=1,stroke=0); c.rect(43,800,24,12,fill=1,stroke=0)
    c.setFillColorRGB(1,1,1); c.setFont('Helvetica-Bold',19); c.drawString(92,807,HOSP)
    c.setFont('Helvetica',9); c.drawString(92,791,'Civil Lines, Dehradun, Uttarakhand  |  Ph: 0135-2000000  |  www.medicare-hospital.in')
    c.saveState(); c.setFillColorRGB(.06,.46,.43); c.setFillAlpha(.14); c.setFont('Helvetica-Bold',24); c.translate(26,420); c.rotate(90); c.drawCentredString(0,0,HOSP.upper()); c.restoreState()
    c.setFillColorRGB(0,0,0); c.setFont('Helvetica-Bold',14); c.drawCentredString(297,743,title)

def stamp(c, cx, cy):
    w = HOSP.upper().split(); top = ' '.join(w[:-1]) or w[0]; bot = (w[-1] + ' - DEHRADUN') if len(w) > 1 else 'DEHRADUN'
    c.saveState(); c.translate(cx, cy); c.rotate(-14); c.setStrokeColorRGB(.08,.2,.7); c.setFillColorRGB(.08,.2,.7); c.setStrokeAlpha(.8); c.setFillAlpha(.8)
    c.setLineWidth(2); c.circle(0,0,52,fill=0); c.setLineWidth(.8); c.circle(0,0,36,fill=0); c.setFont('Helvetica-Bold',7.5)
    for txt, upper in ((top, True), (bot, False)):
        n = len(txt); span = min(160, n * 8.2)
        for k, ch in enumerate(txt):
            f = k / (n - 1) if n > 1 else .5; c.saveState()
            if upper: c.rotate(span/2 - f*span); c.drawCentredString(0,41,ch)
            else: c.rotate(-span/2 + f*span); c.drawCentredString(0,-47,ch)
            c.restoreState()
    c.setLineWidth(5); c.line(-9,0,9,0); c.line(0,-9,0,9); c.restoreState()

def words(n):
    o = ['','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve','Thirteen','Fourteen','Fifteen','Sixteen','Seventeen','Eighteen','Nineteen']
    t = ['','','Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']
    two = lambda x: o[x] if x < 20 else t[x//10] + (' ' + o[x%10] if x % 10 else '')
    three = lambda x: ((o[x//100] + ' Hundred ') if x >= 100 else '') + (two(x % 100) if x % 100 else '')
    n = int(round(n)); cr, n = divmod(n, 10**7); lk, n = divmod(n, 10**5); th, n = divmod(n, 1000)
    parts = [three(v).strip() + ' ' + nm for v, nm in ((cr,'Crore'),(lk,'Lakh'),(th,'Thousand')) if v] + ([three(n).strip()] if n else [])
    return (' '.join(parts) or 'Zero') + ' Rupees Only'

def stamp_sign(c, a):
    p = os.path.join(B, 'static', 'signatures', f"{a['doctor_id']}.png")
    if os.path.exists(p): c.drawImage(p, 385, 112, width=150, height=46, preserveAspectRatio=True, mask='auto')
    else: c.setFont('Times-Italic', 22); c.setFillColorRGB(.1,.1,.45); c.drawString(392,120,a['dname'].replace('Dr. ','')); c.setFillColorRGB(0,0,0)
    c.line(380,108,550,108); c.setFont('Helvetica-Bold',9); c.drawString(380,96,a['dname']); c.setFont('Helvetica',8); c.drawString(380,85,a['dept'] + ' - Authorized Signatory')
    stamp(c, 462, 152)

def sec(c, y, t):
    c.setFillColorRGB(.9,.95,.95); c.rect(40,y-4,515,18,fill=1,stroke=0); c.setFillColorRGB(.06,.46,.43); c.setFont('Helvetica-Bold',10); c.drawString(46,y+1,t.upper()); c.setFillColorRGB(0,0,0)
    return y-24

def kv(c, x, y, k, v):
    c.setFont('Helvetica-Bold',9); c.drawString(x,y,k); c.setFont('Helvetica',9); c.drawString(x+75,y,str(v))

@app.route('/report/<int:aid>')
@auth(pt=True)
def report(aid):
    if not mine(aid): return redirect('/')
    a = q('SELECT a.*, pt.name pname, pt.age, pt.gender, pt.phone, pt.blood, d.name dname, d.dept FROM appointments a JOIN patients pt ON pt.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id WHERE a.id=?', (aid,), one=True)
    seen, latest = set(), []
    for p in q('SELECT * FROM predictions WHERE patient_id=? ORDER BY id DESC', (a['patient_id'],)):
        if p['disease'] not in seen: seen.add(p['disease']); latest.append(p)
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); hdr(c, 'PATIENT MEDICAL REPORT')
    c.setFont('Helvetica',9); c.drawString(40,722,f"Report No: MC-{a['patient_id']:04d}-{aid:04d}"); c.drawRightString(555,722,'Generated: ' + NOW())
    y = sec(c, 695, 'Patient Details')
    kv(c,46,y,'Name:',a['pname']); kv(c,300,y,'Patient ID:',f"MC-{a['patient_id']:04d}"); y -= 16
    kv(c,46,y,'Age / Gender:',f"{a['age']} yrs / {a['gender']}"); kv(c,300,y,'Blood Group:',a['blood']); y -= 16
    kv(c,46,y,'Phone:',a['phone']); y -= 30
    y = sec(c, y, 'Appointment Details')
    kv(c,46,y,'Doctor:',a['dname']); kv(c,300,y,'Department:',a['dept']); y -= 16
    kv(c,46,y,'Date & Time:',f"{a['date']}  {a['time']}"); kv(c,300,y,'Status:',a['status']); y -= 16
    kv(c,46,y,'Reason:',a['reason'] or '-'); y -= 30
    y = sec(c, y, 'Disease Risk Assessment (ML Screening)')
    c.setFont('Helvetica-Bold',9)
    for x, t in [(46,'Disease'),(130,'Vitals Entered'),(420,'Risk'),(480,'Result')]: c.drawString(x,y,t)
    c.line(40,y-4,555,y-4); y -= 18
    if not latest: c.setFont('Helvetica-Oblique',9); c.drawString(46,y,'No disease prediction recorded for this patient yet.'); y -= 18
    for p in latest:
        v = ', '.join(f'{k} {val:g}' for k, val in json.loads(p['vitals']).items()); bad = p['label'] == 'At risk'
        c.setFont('Helvetica-Bold',9); c.drawString(46,y,p['disease'].capitalize()); c.setFont('Helvetica',7.5); c.drawString(130,y,v[:85])
        c.setFont('Helvetica',9); c.drawString(420,y,f"{p['risk']}%"); c.setFillColorRGB(*((.73,.1,.1) if bad else (.08,.5,.2))); c.setFont('Helvetica-Bold',9); c.drawString(480,y,p['label']); c.setFillColorRGB(0,0,0); y -= 18
    y -= 14; y = sec(c, y, 'Prescribed Medicines (from bill)')
    mn = {m['name'] for m in q('SELECT name FROM medicines')}; meds = []
    for b in q('SELECT * FROM bills WHERE appointment_id=?', (aid,)):
        nm, _, qt = b['item'].rpartition(' x')
        if nm in mn and qt.isdigit(): meds.append((nm, qt, b['amount']))
    c.setFont('Helvetica-Bold',9)
    c.drawString(46,y,'Medicine Name')
    c.line(40,y-4,555,y-4); y -= 18
    if not meds: c.setFont('Helvetica-Oblique',9); c.drawString(46,y,'No medicines billed for this appointment.'); y -= 18
    seen = set(); meds = [m for m in meds if not (m[0] in seen or seen.add(m[0]))]
    for i, (nm, qt, am) in enumerate(meds, 1):
        if y < 215: c.showPage(); hdr(c, 'PATIENT MEDICAL REPORT'); y = 700
        c.setFont('Helvetica',9); c.drawString(46,y,f'{i}. {nm}'); y -= 16
    y -= 14
    if y > 260: y = sec(c, y, "Doctor's Remarks"); c.line(46,y,550,y); c.line(46,y-22,550,y-22)
    stamp_sign(c, a)
    c.setFont('Helvetica-Oblique',7.5); c.drawCentredString(297,40,'Computer-generated report. Disease risk is ML-based screening support and NOT a final diagnosis. Please consult your doctor.')
    c.save(); buf.seek(0); return send_file(buf, mimetype='application/pdf', download_name=f'report_{aid}.pdf')

@app.route('/bill/<int:aid>', methods=['POST'])
@auth('admin','receptionist')
def bill(aid):
    q('INSERT INTO bills(appointment_id,item,amount) VALUES(?,?,?)', (aid, request.form['item'], request.form['amount']), commit=True)
    return redirect(request.referrer or '/')

@app.route('/bill/<int:aid>/pdf')
@auth(pt=True)
def bill_pdf(aid):
    if not mine(aid): return redirect('/')
    a = q('SELECT a.*, pt.name pname, pt.age, pt.gender, pt.phone, d.name dname, d.dept FROM appointments a JOIN patients pt ON pt.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id WHERE a.id=?', (aid,), one=True)
    mn = {m['name'] for m in q('SELECT name FROM medicines')}; rows = []
    for b in q('SELECT * FROM bills WHERE appointment_id=?', (aid,)):
        nm, _, qt = b['item'].rpartition(' x'); med = nm in mn and qt.isdigit(); n = int(qt) if med else 1; rate = MED_GST if med else SERVICE_GST
        tax = b['amount'] / (1 + rate / 100); rows.append((nm if med else b['item'], '3004' if med else '9993', n, b['amount'] / n, tax, rate, (b['amount'] - tax) / 2, b['amount']))
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); hdr(c, 'TAX INVOICE'); c.setFont('Helvetica',8); c.drawRightString(555,745,'Original for Recipient')
    c.rect(40,655,255,70); c.rect(300,655,255,70); c.rect(40,605,515,46)
    c.setFont('Helvetica-Bold',9); c.drawString(46,712,'Sold By'); c.drawString(306,712,'Invoice Details'); c.drawString(46,639,'Billed To')
    c.setFont('Helvetica',9)
    for k, t in enumerate([HOSP, 'Civil Lines, Dehradun, Uttarakhand', 'Ph: 0135-2000000', 'GSTIN: ' + HOSP_GSTIN]): c.drawString(46,698-k*13,t)
    for k, t in enumerate([f"Invoice No: INV-{a['patient_id']:04d}-{aid:04d}", f"Date: {a['date']}", f"Appointment ID: {aid}", f"Doctor: {a['dname']} ({a['dept']})"]): c.drawString(306,698-k*13,t)
    c.drawString(46,626,f"{a['pname']}   |   {a['age']} yrs / {a['gender']}   |   Ph: {a['phone']}   |   Patient ID: MC-{a['patient_id']:04d}")
    cols = [(40,25,'Sr'),(65,150,'Description'),(215,45,'HSN/SAC'),(260,28,'Qty'),(288,50,'Rate'),(338,52,'Taxable'),(390,32,'GST%'),(422,42,'CGST'),(464,42,'SGST'),(506,49,'Amount')]
    def thead(y):
        c.setFillColorRGB(.06,.46,.43); c.rect(40,y-18,515,22,fill=1,stroke=0); c.setFillColorRGB(1,1,1); c.setFont('Helvetica-Bold',8)
        for x, w, t in cols: c.drawString(x+3, y-11, t) if t in ('Sr','Description','HSN/SAC') else c.drawRightString(x+w-3, y-11, t)
        c.setFillColorRGB(0,0,0)
    y = 590; thead(y); y -= 18; tt = tc = ga = 0
    for k, (d, hsn, n, rate, tax, gp, half, amt) in enumerate(rows, 1):
        if y < 300: c.showPage(); hdr(c, 'TAX INVOICE'); y = 700; thead(y); y -= 18
        c.setFont('Helvetica',8.5); c.rect(40,y-20,515,20,fill=0)
        vals = [str(k), d[:34], hsn, str(n), f'{rate:,.2f}', f'{tax:,.2f}', f'{gp}%', f'{half:,.2f}', f'{half:,.2f}', f'{amt:,.2f}']
        for (x, w, t), v in zip(cols, vals): c.drawString(x+3, y-14, v) if t in ('Sr','Description','HSN/SAC') else c.drawRightString(x+w-3, y-14, v)
        y -= 20; tt += tax; tc += half; ga += amt
    y -= 14; c.rect(340,y-76,215,76); c.setFont('Helvetica',9)
    for k, (t, v) in enumerate([('Taxable Value', tt), ('CGST', tc), ('SGST', tc)]): c.drawString(346,y-14-k*14,t); c.drawRightString(549,y-14-k*14,f'Rs {v:,.2f}')
    c.setFillColorRGB(.06,.46,.43); c.rect(340,y-76,215,22,fill=1,stroke=0); c.setFillColorRGB(1,1,1); c.setFont('Helvetica-Bold',10); c.drawString(346,y-70,'GRAND TOTAL'); c.drawRightString(549,y-70,f'Rs {ga:,.2f}'); c.setFillColorRGB(0,0,0)
    c.setFont('Helvetica-Bold',9); c.drawString(46,y-14,'Amount in words:'); c.setFont('Helvetica',9); c.drawString(46,y-28,words(ga)[:62])
    c.setFont('Helvetica',7.5); c.drawString(46,y-48,f'Amounts are GST-inclusive. Medicines: {MED_GST}% GST. Health services: exempt ({SERVICE_GST}%).')
    c.setFont('Helvetica-Bold',8); c.drawString(46,140,'Terms & Conditions'); c.setFont('Helvetica',7.5)
    for k, t in enumerate(['1. Medicines once sold will not be taken back.','2. Please check the bill before leaving the counter.','3. Subject to Dehradun jurisdiction.']): c.drawString(46,128-k*11,t)
    stamp(c, 465, 150); c.line(385,98,550,98); c.setFont('Helvetica-Bold',8); c.drawString(385,88,'For ' + HOSP); c.setFont('Helvetica',8); c.drawString(385,78,'Authorised Signatory')
    c.setFont('Helvetica-Oblique',7.5); c.drawCentredString(297,40,'This is a computer-generated invoice.'); c.save(); buf.seek(0)
    return send_file(buf, mimetype='application/pdf', download_name=f'invoice_{aid}.pdf')

@app.route('/patient/<int:pid>/edit', methods=['GET','POST'])
@auth('admin','receptionist', pt=True)
def edit_patient(pid):
    if session['role'] == 'patient' and session['pid'] != pid: return redirect(f"/patient/{session['pid']}")
    if request.method == 'POST':
        f = request.form; q('UPDATE patients SET name=?,age=?,gender=?,phone=?,blood=? WHERE id=?', (f['name'], f['age'], f['gender'], f['phone'], f['blood'], pid), commit=True)
        flash('Patient details updated.'); return redirect(f'/patient/{pid}')
    return render_template('edit_patient.html', p=q('SELECT * FROM patients WHERE id=?', (pid,), one=True))

@app.route('/medicines', methods=['GET','POST'])
@auth()
def medicines():
    if request.method == 'POST' and session['role'] == 'admin':
        f = request.form
        if f.get('action') == 'price': q('UPDATE medicines SET price=? WHERE id=?', (float(f['price']), f['id']), commit=True)
        else:
            from werkzeug.utils import secure_filename
            img, fl = 'default.svg', request.files.get('photo')
            if fl and fl.filename: img = secure_filename(f['name']) + '_' + secure_filename(fl.filename); fl.save(os.path.join(B, 'static', 'medicines', img))
            q('INSERT INTO medicines(name,disease,price,image) VALUES(?,?,?,?)', (f['name'], f['disease'], float(f['price']), img), commit=True)
        flash('Medicine saved.'); return redirect('/medicines')
    return render_template('medicines.html', rows=q('SELECT * FROM medicines ORDER BY name'), diseases=list(FEATS))

@app.route('/bill/<int:aid>/medicine', methods=['POST'])
@auth('admin','receptionist')
def bill_med(aid):
    m = q('SELECT * FROM medicines WHERE id=?', (request.form['mid'],), one=True); n = max(1, int(request.form.get('qty', 1)))
    q('INSERT INTO bills(appointment_id,item,amount) VALUES(?,?,?)', (aid, f"{m['name']} x{n}", m['price'] * n), commit=True); return redirect(request.referrer or '/')

@app.route('/revenue')
@auth('admin','receptionist')
def revenue():
    kw = '%' + request.args.get('s', '') + '%'
    rows = q("""SELECT a.id aid, a.date, a.reason, pt.id pid, pt.name, pt.age, pt.gender, pt.phone, d.name dname,
        (SELECT GROUP_CONCAT(DISTINCT disease) FROM predictions WHERE patient_id=pt.id) diseases,
        COALESCE((SELECT SUM(amount) FROM bills WHERE appointment_id=a.id),0) total
        FROM appointments a JOIN patients pt ON pt.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id
        WHERE pt.name LIKE ? OR pt.phone LIKE ? ORDER BY a.date DESC, a.id DESC""", (kw, kw))
    tot = q('SELECT COALESCE(SUM(amount),0) s, COUNT(DISTINCT appointment_id) n FROM bills', one=True)
    return render_template('revenue.html', rows=rows, tot=tot, s=request.args.get('s', ''))

@app.route('/doctor/<int:did>/signature', methods=['POST'])
@auth('admin')
def doc_sig(did):
    fl = request.files.get('sig')
    if fl and fl.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        os.makedirs(os.path.join(B, 'static', 'signatures'), exist_ok=True); fl.save(os.path.join(B, 'static', 'signatures', f'{did}.png')); flash('Signature uploaded.')
    else: flash('Please upload a PNG or JPG image.')
    return redirect('/doctors')

POSTS = ['Doctor','Nurse','Receptionist','Pharmacist','Lab Technician','Ward Boy','Admin Staff','Other']

@app.route('/staff', methods=['GET','POST'])
@auth('admin')
def staff():
    if request.method == 'POST':
        f = request.form
        if f.get('action') == 'add':
            q('INSERT INTO staff(name,post,department,salary,phone,joined) VALUES(?,?,?,?,?,?)', (f['name'], f['post'], f['dept'], float(f['salary']), f['phone'], f['joined']), commit=True)
            if f['post'] == 'Doctor': q('INSERT INTO doctors(name,dept,phone) VALUES(?,?,?)', (f['name'], f['dept'], f['phone']), commit=True)
            flash('Staff member added.')
        else:
            q('UPDATE staff SET salary=?, status=? WHERE id=?', (float(f['salary']), f['status'], f['id']), commit=True); flash('Staff updated.')
        return redirect('/staff')
    summ = q("SELECT post, COUNT(*) n, SUM(salary) total, AVG(salary) avg FROM staff WHERE status='Active' GROUP BY post ORDER BY n DESC")
    tot = q("SELECT COUNT(*) n, COALESCE(SUM(salary),0) s FROM staff WHERE status='Active'", one=True)
    kw = '%' + request.args.get('s', '') + '%'; post = request.args.get('post', '')
    sql = 'SELECT * FROM staff WHERE (name LIKE ? OR department LIKE ? OR phone LIKE ?)'; args = [kw, kw, kw]
    if post: sql += ' AND post=?'; args.append(post)
    return render_template('staff.html', rows=q(sql + ' ORDER BY post, name', tuple(args)), summ=summ, tot=tot, posts=POSTS, today=dt.date.today().isoformat(), s=request.args.get('s', ''), post=post)

@app.route('/users', methods=['GET','POST'])
@auth('admin')
def users():
    if request.method == 'POST':
        f = request.form
        if f.get('action') == 'toggle':
            u = q('SELECT * FROM users WHERE id=?', (f['id'],), one=True)
            if u['username'] == session['user']: flash("You can't disable your own account.")
            else: q('UPDATE users SET active=? WHERE id=?', (0 if u['active'] != 0 else 1, u['id']), commit=True)
        elif len(f['password']) < 6: flash('Password must be at least 6 characters.')
        else:
            try: q('INSERT INTO users(username,password,role,name) VALUES(?,?,?,?)', (f['username'], generate_password_hash(f['password']), f['role'], f['name']), commit=True); flash('Account created.')
            except sqlite3.IntegrityError: flash('Username already exists.')
        return redirect('/users')
    return render_template('users.html', rows=q('SELECT * FROM users ORDER BY role, username'))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        f = request.form
        if len(f['password']) < 6: flash('Password must be at least 6 characters.'); return redirect('/signup')
        if q('SELECT 1 FROM users WHERE username=?', (f['username'],), one=True): flash('Username already taken.'); return redirect('/signup')
        pid = q('INSERT INTO patients(name,age,gender,phone,blood,created) VALUES(?,?,?,?,?,?)', (f['name'], f['age'], f['gender'], f['phone'], f['blood'], NOW()), commit=True)
        q('INSERT INTO users(username,password,role,name,patient_id) VALUES(?,?,?,?,?)', (f['username'], generate_password_hash(f['password']), 'patient', f['name'], pid), commit=True)
        session.update(user=f['username'], role='patient', name=f['name'], pid=pid); return redirect(f'/patient/{pid}')
    return render_template('signup.html')

@app.route('/password', methods=['GET','POST'])
@auth(pt=True)
def password():
    if request.method == 'POST':
        f = request.form; u = q('SELECT * FROM users WHERE username=?', (session['user'],), one=True)
        if not check_password_hash(u['password'], f['old']): flash('Current password is wrong.')
        elif len(f['new']) < 6 or f['new'] != f['confirm']: flash('New passwords must match and be at least 6 characters.')
        else: q('UPDATE users SET password=? WHERE id=?', (generate_password_hash(f['new']), u['id']), commit=True); flash('Password changed.')
        return redirect('/password')
    return render_template('password.html')

if __name__ == '__main__':
    init()
    import socket
    try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close()
    except Exception: ip = '127.0.0.1'
    print(f'\nLaptop par kholo : http://127.0.0.1:5000\nMOBILE par kholo : http://{ip}:5000  (same WiFi)\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
