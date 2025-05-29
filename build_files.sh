#!/bin/bash
# Script para recopilar archivos estáticos para Django en Vercel
pip install -r requirements.txt
python manage.py collectstatic --noinput 