from facebook_scraper import get_posts
import csv

with open('facebookScrapper.csv', 'a+') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=['POSTS'])
    csvwriter.writeheader()
    result={'POSTS':''}
    for post in get_posts('ImranKhanOfficial', pages=1):
        result['POSTS'] = post['text']+'\n'
        csvwriter.writerow(result)
