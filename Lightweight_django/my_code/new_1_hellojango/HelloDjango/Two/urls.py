from django.conf.urls import url

from Two import views

urlpatterns ={
    url(r'^index/',views.index),
    url(r'^addstudent/',views.add_student),
    url(r'^getstudent/',views.get_student),
    url(r'^updatestudent/',views.update_student),
    url(r'^deleteStudent/',views.delete_student),
}