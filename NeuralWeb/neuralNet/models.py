from django.db import models

# Create your models here.
class DigitData (models.Model):
	digit = models.CharField(max_length=50)
	pixelMap = models.CharField(max_length=1000)