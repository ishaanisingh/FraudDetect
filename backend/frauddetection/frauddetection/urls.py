"""
URL configuration for frauddetection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# frauddetection/urls.py

from django.contrib import admin
from django.urls import path
from app import views  # Import your views from the 'app' folder

urlpatterns = [
    # 1. Admin Site
    path('admin/', admin.site.urls),
    
    # 2. LISTING AND PREDICTION (POST)
    path('api/transactions/', views.transaction_list_create, name='transaction_list_create'),
    
    # 3. STATS / DASHBOARD API
    path('api/validation-data/', views.validation_data, name='validation_data'),
    
    # 4. SINGLE TRANSACTION DETAIL
    path('api/transactions/<int:pk>/', views.transaction_detail, name='transaction_detail'),
    
    # NOTE: The line path('api/predict/', views.predict, name='predict'), is the source of the crash and must be deleted.
]