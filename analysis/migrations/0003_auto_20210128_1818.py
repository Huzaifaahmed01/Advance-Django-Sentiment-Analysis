# Generated by Django 3.1.5 on 2021-01-28 18:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0002_sentimentreview'),
    ]

    operations = [
        migrations.RenameField(
            model_name='sentimenttext',
            old_name='residentialAddress',
            new_name='submittedText',
        ),
    ]