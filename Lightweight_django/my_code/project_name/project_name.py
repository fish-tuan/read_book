#01
from django.http import HttpResponse
from django.conf.urls import url
from django.conf import settings
import sys
from django.core.management import execute_from_command_line
import os

DEBUG =  os.environ.get('DEBUG','on') == 'on'
SECRET_KEY = os.environ.get('SECRET_KEY',os.urandom(32))
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS','localhost').split(',')

settings.configure(
    DEBUG=DEBUG,
    SECRET_KEY=SECRET_KEY,
    ALLOWED_HOSTS = ALLOWED_HOSTS,
    ROOT_URLCONF = __name__,
    MIDDLEWARE_CLASSES =(
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ),
)




def index(request):
    return HttpResponse('Hello World,I love you')

urlpatterns =( url(r'^$',index),)


if __name__ =='__main__':
    execute_from_command_line(sys.argv)