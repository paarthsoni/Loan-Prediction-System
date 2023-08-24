from django.db import models


class loanpredictiondata(models.Model):
    gender = models.CharField(null=True, max_length=1024)
    marital_status = models.CharField(null=True, max_length=1024)
    dependents = models.CharField(null=True, max_length=1024)
    education = models.CharField(null=True, max_length=1024)
    self_employed = models.CharField(null=True, max_length=1024)
    applicantincome = models.CharField(null=True, max_length=1024)
    coapplicantincome = models.CharField(null=True, max_length=1024)
    loan_status = models.CharField(null=True, max_length=1024)
    loan_amount = models.CharField(null=True, max_length=1024)

    def __str__(self):
        return self.loan_status
