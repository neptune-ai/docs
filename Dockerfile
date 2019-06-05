FROM nginx:stable

COPY docs/_build/html/ /usr/share/nginx/html

RUN mkdir /app

ADD docker/nginx.conf /etc/nginx/conf.d/default.conf
ADD docker/run_app.sh /app

CMD /app/run_app.sh
