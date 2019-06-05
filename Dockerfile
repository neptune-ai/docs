FROM nginx:stable

COPY docs/_build/html/ /usr/share/nginx/html

ADD docker/nginx.conf /etc/nginx/conf.d/default.conf
