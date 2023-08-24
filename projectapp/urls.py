from django.contrib import admin
from django.urls import include, path
from projectapp import views
urlpatterns = [
    path('', views.index, name="home"),
    path('prediction',views.loan_prediction,name='prediction'),
]
