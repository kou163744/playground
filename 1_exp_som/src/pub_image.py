import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import sys

def publish_image():
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('/yamao/image_topic', Image, queue_size=10)
    bridge = CvBridge()
    args = sys.argv
    
    # JPEG画像を読み込み
    img = cv2.imread(f'../io/data/level2_image/rgb_input{args[1]}.jpg')
    
    if img is None:
        rospy.logerr("Image not found or unable to open")
        return

    rate = rospy.Rate(10) # 10Hz

    while not rospy.is_shutdown():
        # OpenCV画像をROSのImageメッセージに変換
        image_message = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        
        # トピックとして送信
        image_pub.publish(image_message)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
