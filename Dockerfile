FROM nginx:stable

COPY docs/_build/html/ /usr/share/nginx/html

ADD docker/nginx.conf /etc/nginx/conf.d/default.conf

ENV WORKDIR /app

RUN mkdir /app

WORKDIR /app

ADD docker/generate_settings.sh /app
ADD docker/entrypoint.sh /app

ENTRYPOINT ["/app/entrypoint.sh"]

