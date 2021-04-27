from django.urls import path, include
from django.conf.urls import handler404
from django.conf.urls import url, include
from django.contrib import admin
from bonge.resources import NoteResource
note_resource = NoteResource()


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bonge.urls')),
    url(r'^bonge/', include(note_resource.urls)),
]


handler404 = 'bonge.views.handler404'
