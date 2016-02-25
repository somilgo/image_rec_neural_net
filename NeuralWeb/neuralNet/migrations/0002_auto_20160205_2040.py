# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('neuralNet', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='digitdata',
            name='digit',
            field=models.CharField(max_length=50),
        ),
    ]
