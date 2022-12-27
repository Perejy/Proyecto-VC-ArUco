# Proyecto-VC-ArUco

Proyecto para la asignatura de visión por computador de la universidad de Málaga.

Este trabajo se centra principalmente en la investigación y compresión de los marcadores aruco.

## Librerias a instalar 

Recomiendo instalar un entorno virtual para la instalación de librerias, en mi caso hice uso de venv.
Los comandos para la instalación del entorno virtual y las librerias son los siguientes:
```
python3 -m venv venv-name
venv-name\Scripts\activate  
python3 -m pip install --upgrade pip
pip install numpy opencv-contrib-python opencv-python scipy matplotlib
```

### Nota
El proyecto está configurado para usarlo con una cámara externa, para ello usé la aplicación *droidcam* disponible en android con la que enlacé la cámara del movil al ordenador. Por otro lado, tambien recomendaría modificar las imagenes de calibrado por unas tomadas por la cámara que se vaya a usar.

Otra opción es usar una cámara interna del propio ordenador como la de los portátiles, para ello modificar la siguiente linea de código:
```
cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa.
```
