# Generated by Django 3.2 on 2021-04-27 22:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bonge', '0003_alter_note_reference_allele'),
    ]

    operations = [
        migrations.AlterField(
            model_name='note',
            name='Affected_Allele',
            field=models.TextField(max_length=15),
        ),
        migrations.AlterField(
            model_name='note',
            name='Reference_Allele',
            field=models.TextField(max_length=15),
        ),
    ]
