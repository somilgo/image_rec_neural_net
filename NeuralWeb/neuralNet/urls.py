from django.conf.urls import include, url, patterns
from django.contrib import admin
from django.conf import settings

from . import views

urlpatterns = [
	url(r'^$', views.main, name='main'),
	url(r'^run_digit_network/$', views.run_digit_network, name='run')
]