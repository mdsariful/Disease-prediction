
from django.contrib import admin
from django.urls import path, include
from bonge import views



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bonge.urls')),
    
    
]


