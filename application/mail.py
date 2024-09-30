import os
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()

subject = '''Thank You for Your Participation! Discover Our Exclusive Offers!'''
message = '''
We extend our heartfelt thanks for participating in our recent market survey. Your valuable feedback has been instrumental in helping us understand your preferences better and tailoring our products to meet your needs.
We are excited to introduce you to our range of products, perfectly suited to enhance your lifestyle. As a token of our appreciation, we have curated some exclusive offers just for you.'''
end_msg = '''
Best regards,
Anmol Mani Dubey
Customer Relations Manager
Flipkart

P.S. Don't miss out on these exclusive offers! Visit our website www.flipkart.com to explore our products and make the most of these discounts.'''
clusterA = '''
Thank you for your continued support! As a valued customer, we have a special offer just for you. Enjoy an exclusive 15% discount on any of our products using the code YOUNG15. We believe this will perfectly suit your dynamic lifestyle and help you achieve more with ease.
'''
clusterB = '''
Thank you for being a valued customer! We understand your need for a product that combines practicality with modern design. Enjoy a 20% discount on any of our products with the code FAMILY20. This offer is designed to cater to both you and your family's needs, making life a little easier and more enjoyable.
'''
clusterC = '''
Thank you for your loyalty! Balancing personal and professional life is important, and we have the perfect offer to help you with that. Use the code BALANCE10 to enjoy a 10% discount on any of our products. Our range of products is crafted to provide reliability and comfort, ensuring you stay ahead in your endeavors.
'''
clusterD = '''
Thank you for your trust and support! We appreciate your sophisticated taste and are delighted to offer you a special discount. Use the code ELDER25 to receive a 25% discount on any of our products. Our products offer premium features that cater to your advanced needs, ensuring you get the best in performance and style.
'''

def send_email(to_email, name, prediction):
    if prediction == 'cluster A':
        cluster = clusterA
    elif prediction == 'cluster B':
        cluster = clusterB
    elif prediction == 'cluster C':
        cluster = clusterC
    else:
        cluster = clusterD

    from_email = os.getenv("EMAIL")
    from_password = os.getenv("PASSWORD")

    body = f'''
Dear {name},
{message}
{cluster}
{end_msg}
    '''

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
