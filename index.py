import sys
sys.path.append('.modules')
from api.videohash import VideoHash
videohash1 = VideoHash(url="https://dawcqwjlx34ah.cloudfront.net/dcc7ba87-b10f-4a21-9352-290f433292fe_1153547682117989.mp4")
print(videohash1)