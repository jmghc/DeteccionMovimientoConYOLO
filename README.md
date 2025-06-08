# 📹 Sistema de Detección de Personas con YOLOv3 y Alerta por Correo Electrónico

Este proyecto consiste en un sistema inteligente de videovigilancia capaz de detectar personas en tiempo real mediante la cámara de un dispositivo. Al identificar presencia humana, el sistema captura una imagen y la envía automáticamente por correo electrónico como alerta. La detección se realiza mediante el modelo de deep learning **YOLOv3 (You Only Look Once)** y técnicas de visión por computadora con OpenCV.

---

## 🎓 Contexto académico

Este trabajo ha sido desarrollado en el marco de la **asignatura de Programación de Inteligencia Artificial**, dentro del **Curso de Especialización en Inteligencia Artificial y Big Data**.

El objetivo principal es poner en práctica técnicas de visión artificial y automatización inteligente con Python, integrando modelos de detección preentrenados, análisis en tiempo real y mecanismos de notificación automática.

---

## 🧠 Tecnologías y conceptos aplicados

- 📦 Python 3
- 🎯 Detección de objetos con **YOLOv3**
- 👁️ Visión por computadora con **OpenCV**
- 📩 Envío automático de correos con `smtplib`
- 📷 Captura y procesamiento de vídeo en tiempo real
- 🛡️ Automatización de alertas para vigilancia y seguridad

---
## 📥 Descarga del modelo YOLOv3

⚠️ El archivo `yolov3.weights` no está incluido en este repositorio debido a su gran tamaño (> 100 MB).  
Puedes descargarlo manualmente desde este enlace y colocarlo en la carpeta `Modelo/`:

🔗 [Descargar yolov3.weights desde Google Drive](https://drive.google.com/file/d/16INayobpSwe6du28hKwQsJokOZ1fHyHh/view?usp=sharing)

Una vez descargado, colócalarlo en carpeta Modelo:
## Configuración del correo.
En el main colocar correo y contraseña para enviar captura de movimiento detectado.


