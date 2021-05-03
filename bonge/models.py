from django.db import models
class Note(models.Model):

    RsID = models.CharField(max_length=200)
    Affected_Allele = models.TextField(max_length=100)
    Reference_Allele = models.TextField( max_length=100)
    
    
    
    
    def __str__(self):
        return '{}, {}, {}'.format(self.RsID, self.Affected_Allele, self.Reference_Allele)
        
