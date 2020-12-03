from facebook_scraper import get_posts

for post in get_posts('ImranKhanOfficial', pages=2):
    print(post['text'])
