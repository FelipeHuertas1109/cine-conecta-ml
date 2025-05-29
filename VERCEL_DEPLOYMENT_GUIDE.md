# Guía de Despliegue en Vercel

## Configuración de Variables de Entorno

Para desplegar correctamente este proyecto en Vercel, necesitas configurar las siguientes variables de entorno en el panel de control de Vercel:

### Variables de Entorno Esenciales

```
DEBUG=False
SECRET_KEY=genera-una-clave-secreta-larga-y-segura-para-produccion
SENTIMENT_MODEL_PATH=svr_spacy.joblib
```

## Problemas Comunes y Soluciones

### 1. Error: FUNCTION_INVOCATION_FAILED

Este error generalmente ocurre por alguna de estas razones:

- **Modelo ML demasiado grande**: El archivo `svr_spacy.joblib` (59MB) es demasiado grande para las funciones serverless de Vercel.
  
  **Soluciones:**
  - Utiliza un modelo más pequeño (versión _sm de spaCy en lugar de _md)
  - Aloja el modelo en un servicio de almacenamiento externo (AWS S3, Google Cloud Storage)
  - Divide la aplicación: API en Vercel, procesamiento ML en otro servicio

- **Tiempo de ejecución excedido**: Las funciones de Vercel tienen un límite de tiempo.
  
  **Solución:** Optimiza el código para que la respuesta sea más rápida.

### 2. Manejo de Archivos Estáticos

Asegúrate de que el script `build_files.sh` tenga permisos de ejecución antes de subir a Vercel:

```bash
chmod +x build_files.sh
```

## Opciones Alternativas de Despliegue

Si continúas teniendo problemas con Vercel debido al tamaño del modelo ML, considera:

1. **Dividir la aplicación**:
   - Frontend/API en Vercel
   - Modelo ML en un servicio especializado (AWS Lambda, Google Cloud Functions)

2. **Plataformas alternativas**:
   - Heroku
   - PythonAnywhere
   - DigitalOcean App Platform

## Prueba Local con Configuración de Producción

Para probar localmente con la configuración de producción, crea un archivo `.env` con:

```
DEBUG=False
SECRET_KEY=tu-clave-secreta
SENTIMENT_MODEL_PATH=svr_spacy.joblib
```

Y ejecuta:

```bash
python manage.py runserver
``` 