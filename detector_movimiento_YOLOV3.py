import cv2
import numpy as np
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# --------------- CONFIGURACIÓN DEL CORREO ELECTRÓNICO (Poner correo al que se desea enviar la captura)---------------
SMTP_SERVER = "smtp.gmail.com"  # Servidor SMTP de Gmail
SMTP_PORT = 587  # Puerto para TLS
EMAIL_ADDRESS = ""  # Tu dirección de correo
EMAIL_PASSWORD = ""  # Tu contraseña de correo
DESTINATION_EMAIL = ""  # Correo destino
#Función para enviar capturas de detección de movimiento al email
def send_email(subject, body, attachment_path=None):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = DESTINATION_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Adjuntar la captura si existe
        if attachment_path:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(attachment_path)}",
            )
            msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, DESTINATION_EMAIL, msg.as_string())
        server.quit()
        print("Correo enviado exitosamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")
# --------------- CARGAR MODELO DE YOLO v3 ---------------
config = "Modelo/yolov3.cfg"
weights = "Modelo/yolov3.weights"
LABELS = open("Modelo/coco.names").read().strip().split("\n")
TARGET_CLASS = "person" #Solo para obtener a las personas del coco.names
person_class_id = LABELS.index(TARGET_CLASS) if TARGET_CLASS in LABELS else -1
if person_class_id == -1:
    print(f"Error: La clase '{TARGET_CLASS}' no esta en COCO.")
    exit()

colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(config, weights)

# --------------- CONFIGURACIÓN DE LAS CÁMARAS ---------------
cap = cv2.VideoCapture(0) #Ejemplo con mi camara
#cap = cv2.VideoCapture('https://plataforma.caceres.es/streaming/santamaria')
#cap = cv2.VideoCapture('https://plataforma.caceres.es/streaming/sanmateo')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Variables de control para el correo
time_threshold = 60
last_email_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    persona_detectada = False
    detectiones = []

    # --------------- DETECCIÓN DE MOVIMIENTO ---------------
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detectiones.append((x, y, w, h))

    # --------------- DETECCIÓN DE PERSONAS CON YOLO ---------------
    if detectiones:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        ln = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classID == person_class_id and confidence > 0.5:
                    persona_detectada = True
                    box = detection[:4] * np.array([width, height, width, height])
                    (x_center, y_center, w, h) = box.astype("int")
                    x = int(x_center - (w / 2))
                    y = int(y_center - (h / 2))
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if idx is not None and len(idx) > 0:
            detectiones = []
            for i in idx.flatten():
                x, y, w, h = boxes[i]
                color = [int(c) for c in colors[classIDs[i]]]
                text = f"Persona: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --------------- LÓGICA DE ALERTA Y ENVÍO DE CORREO ---------------
    if persona_detectada and (time.time() - last_email_time) > time_threshold:
        captura_path = f"captura_{int(time.time())}.jpg"
        cv2.imwrite(captura_path, frame)
        send_email("¡Alerta! Persona detectada", "Se ha detectado una persona en la camara.", captura_path)
        os.remove(captura_path)  # Eliminar la captura después de enviarla
        last_email_time = time.time()

    texto_estado = "¡Alerta! Se detecta movimiento de persona" if persona_detectada else "No se detecta movimiento"
    color_estado = (0, 0, 255) if persona_detectada else (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_estado, 2)

    cv2.imshow("Deteccion", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
