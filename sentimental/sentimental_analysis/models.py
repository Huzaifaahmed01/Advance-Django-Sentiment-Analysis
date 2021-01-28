from django.db import models


# Create your models here.
class sentimentText(models.Model):
    residentialAddress = models.TextField(max_length=512)

class sentimentReview(models.Model):
    SENT_TYPE = [
        ('POS', 'Positive'),
        ('NEG', 'Negative')
    ]
    sentimentType = models.CharField(max_length=3, choices=SENT_TYPE, blank=False)
    EMOTION_TYPE = [
        ('NET', 'Neutral'),
        ('EMP', 'Empty'),
        ('SAD', 'Sadness'),
        ('ENT', 'Enthusiasm'),
        ('WOR', 'Worry'),
        ('SUR', 'Surprise'),
        ('LOV', 'Love'),
        ('FUN', 'Fun'),
        ('HAT', 'Hate'),
        ('HAP', 'Happiness'),
        ('BOR', 'Boredom'),
        ('REL', 'Relief'),
        ('ANG', 'Anger')
        ]
    emotionType = models.CharField(max_length=3, choices=EMOTION_TYPE, blank=False)
    textReview = models.ForeignKey(sentimentText, on_delete=models.CASCADE)
