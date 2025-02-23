FROM python:3.9

RUN useradd -m myuser

# ************************* #
## change to port 80 when running on AWS EC2 ##
EXPOSE 5000
# ************************* #

RUN mkdir -p /app

# ************************* #
# when copying directories must create directories in docker container as well #
COPY static /app/static
COPY templates /app/templates
# ************************* #

COPY main.py /app
COPY test_app.py /app
COPY model.sav /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Set ownership and permissions for the non-root user
RUN chown -R myuser:myuser /app
RUN chmod -R 755 /app

ENTRYPOINT pytest -v && python /app/main.py
