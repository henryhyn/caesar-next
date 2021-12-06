from tensorflow.contrib import predictor

from get_started.prepare_data import _int64_feature, _bytes_feature, _float_feature
from tutorials.job_card_examples import examples
from utils import tf_settings
from utils.time_util import timeit_no_args


class JobCardService:
    def __init__(self):
        self.tf = tf_settings()
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.session = self.tf.Session(graph=self.graph)
            with self.session.as_default():
                self.model = predictor.from_saved_model(export_dir='/data/models/job_card/20211208140343/1638948439')
        self._warm_up()

    def _warm_up(self):
        keys = examples.keys()
        records = [dict(zip(keys, vals)) for vals in zip(*(examples[k] for k in keys))]
        inputs = [self.serialize_example(**record) for record in records]
        # [0.09411349, 0.5888527, 0.18392204, 0.37702212, 0.7794226]
        self.predict(inputs)
        self.predict(inputs)

    @timeit_no_args
    def predict(self, inputs):
        with self.graph.as_default():
            with self.session.as_default():
                predictions = self.model({'inputs': inputs})
        return predictions['scores'][:, 1]

    def serialize_example(self, **kwargs):
        feature = {
            'from_create': _float_feature([kwargs['from_create']]),
            'avg_process_rate': _float_feature([kwargs['avg_process_rate']]),
            'avg_process_sec': _float_feature([kwargs['avg_process_sec']]),
            'day_salary_min': _float_feature([kwargs['day_salary_min']]),
            'day_salary_max': _float_feature([kwargs['day_salary_max']]),
            'gender': _bytes_feature([kwargs['gender'].encode('utf-8')]),
            'school_type': _bytes_feature([kwargs['school_type'].encode('utf-8')]),
            'edu_level': _bytes_feature([kwargs['edu_level'].encode('utf-8')]),
            'i_edu_level': _int64_feature([kwargs['i_edu_level']]),
            'honor_level': _int64_feature([kwargs['honor_level']]),
            'job_city': _bytes_feature([kwargs['job_city'].encode('utf-8')]),
            'live_place': _bytes_feature([kwargs['live_place'].encode('utf-8')]),
            'work_want_place': _bytes_feature([kwargs['work_want_place'].encode('utf-8')]),
            'career_job1_1': _bytes_feature([kwargs['career_job1_1'].encode('utf-8')]),
            'career_job1_2': _bytes_feature([kwargs['career_job1_2'].encode('utf-8')]),
            'career_job1_3': _bytes_feature([kwargs['career_job1_3'].encode('utf-8')]),
            'career_job_id': _int64_feature([kwargs['career_job_id']]),
            'company_id': _int64_feature([kwargs['company_id']]),
            'job_id': _int64_feature([kwargs['job_id']])
        }
        example = self.tf.train.Example(features=self.tf.train.Features(feature=feature))
        return example.SerializeToString()


if __name__ == '__main__':
    JobCardService()
