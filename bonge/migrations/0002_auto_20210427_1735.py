# Generated by Django 3.2 on 2021-04-27 09:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bonge', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='note',
            old_name='body',
            new_name='Affected_Allele',
        ),
        migrations.RenameField(
            model_name='note',
            old_name='created_at',
            new_name='Reference_Allele',
        ),
        migrations.RenameField(
            model_name='note',
            old_name='title',
            new_name='rsID',
        ),
    ]
