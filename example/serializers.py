from rest_framework import serializers

class ReviewSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=10_000)
