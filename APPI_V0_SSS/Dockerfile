FROM python:3.10

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo
WORKDIR /opt/api-dosis

# Copiar archivos del proyecto al contenedor
ADD ./ /opt/api-dosis/

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r /opt/api-dosis/requirements.txt

# Hacer el directorio ejecutable
RUN chmod +x /opt/api-dosis/run.sh

# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

# Ejecutar el contenedor como el usuario api-user
USER api-user

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8001

# Comando para ejecutar la aplicación
CMD ["bash", "./run.sh"]
