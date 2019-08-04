import sys
from django.conf import settings
import os

BASE_DIR = os.path.dirname(__file__)

settings.configure(
    DEBUG=True,
    SECRET_KEY='jv_4#hoaqwig2gu!eg#^ozptd*a@88u(aasv7z!7xt^5(*i&k',
    ROOT_URLCONF='sitebuilder.urls',
    MIDDLEWARE_CLASSES=(),
    INSTALLED_APPS=(
        'django.contrib.staticfiles',
        'sitebuilder'
    ),
    TEMPLATES=(
        {
            'BACKEND':'django.template.backends.django.DjangoTemplates',
            'DIRS':[],
            'APP_DIRS':True,
        },
    ),
    STATIC_URL='/static/',
    SITE_PAGES_DIRECTORY= os.path.join(BASE_DIR,'pages')
)

if __name__=="__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)