import numpy as np
import pandas as pd

examples = {
    'from_create': [1991, 3971, 1565, 1080, 2498],
    'avg_process_rate': [99, 99, 91, 29, 94],
    'avg_process_sec': [30524, 34752, 405261, 88789, 46393],
    'day_salary_min': [0, 400, 250, 200, 0],
    'day_salary_max': [9999999, 500, 300, 250, 9999999],
    'gender': ['UK', '男', '女', 'UK', '女'],
    'school_type': ['一本', '211', '211', '其他', '二本'],
    'edu_level': ['硕士', '本科', '硕士', '本科', '本科'],
    'i_edu_level': [5000, 5000, 5000, 5000, 5000],
    'honor_level': [3, 1, 5, 4, 3],
    'job_city': ['北京,广州,深圳,杭州,南京,成都,厦门,武汉,西安,鄂尔多斯,上海,邯郸,本溪,长治',
                 '北京,深圳,杭州,南京,成都,厦门,武汉,西安,广州,赤峰,上海', '北京', '北京', '北京,上海,广州,南京,郑州'],
    'live_place': ['UK', 'UK', 'UK', 'UK', '广东身广州市天河区'],
    'work_want_place': ['北京,上海,深圳,广州,杭州', 'UK', '北京', '郑州,杭州,上海,北京,武汉,西安,成都,南京,厦门,深圳', '广州'],
    'career_job1_1': ['技术（软件）/信息技术类', '技术（软件）/信息技术类', '技术（软件）/信息技术类', '技术（软件）/信息技术类', '技术（软件）/信息技术类'],
    'career_job1_2': ['前端开发', '人工智能', '后端开发', '前端开发', '数据'],
    'career_job1_3': ['前端工程师', '自然语言处理', 'java工程师', 'web前端', '数据分析师'],
    'career_job_id': [11022, 11006, 11003, 11235, 11027],
    'company_id': [665, 665, 138, 5604, 2044],
    'job_id': [53943, 32904, 58357, 63415, 46079]
}


def sample():
    df = pd.read_csv('/data/datasets/job_card_data_20211116_20211206_android_intern.tsv', sep='\t').fillna('UK')
    data = df[df.apply(lambda row: row['y'] == 1 and np.random.uniform() < 0.0001, axis=1)].head()
    for key, val in dict(data).items():
        print(f"'{key}': {list(val)},")


if __name__ == '__main__':
    sample()
