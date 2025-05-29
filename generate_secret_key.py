#!/usr/bin/env python
"""
Script para generar una clave secreta segura para Django.
Ejecuta este script para obtener una nueva clave secreta que puedes usar en producción.
"""
import secrets
import string

def generate_secret_key(length=50):
    """Genera una clave secreta segura para Django."""
    chars = string.ascii_letters + string.digits + string.punctuation
    # Eliminar caracteres que podrían causar problemas en las variables de entorno
    chars = chars.replace("'", "").replace('"', "").replace('\\', "")
    return ''.join(secrets.choice(chars) for _ in range(length))

if __name__ == "__main__":
    secret_key = generate_secret_key()
    print("\nClave secreta generada para Django:")
    print("-" * 60)
    print(secret_key)
    print("-" * 60)
    print("\nCopia esta clave y úsala como valor para SECRET_KEY en producción.")
    print("Agrega esta clave a tus variables de entorno en Vercel.") 