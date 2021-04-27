from django.db import models
class Note(models.Model):
    rsID = models.CharField(max_length=200)
    Affected_Allele = models.TextField()
    Reference_Allele = models.TextField()
def __str__(self):
        return '%s %s' % (self.rsID, self.Affeted_Allele)