from app import app
from flask import render_template, request
from predict import predict_sentiments
from YTComments import get_video_comments
from flask_cors import CORS
import requests

    