import argparse
from os import makedirs, environ
from os.path import abspath, exists, join
from modlr import numerai, util, logs
import logging

dataset = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
column_names = ['id', 'era', 'data_type', 'feature_intelligence1', 'feature_intelligence2', 'feature_intelligence3', 'feature_intelligence4', 'feature_intelligence5', 'feature_intelligence6', 'feature_intelligence7', 'feature_intelligence8', 'feature_intelligence9', 'feature_intelligence10', 'feature_intelligence11', 'feature_intelligence12', 'feature_charisma1', 'feature_charisma2', 'feature_charisma3', 'feature_charisma4', 'feature_charisma5', 'feature_charisma6', 'feature_charisma7', 'feature_charisma8', 'feature_charisma9', 'feature_charisma10', 'feature_charisma11', 'feature_charisma12', 'feature_charisma13', 'feature_charisma14', 'feature_charisma15', 'feature_charisma16', 'feature_charisma17', 'feature_charisma18', 'feature_charisma19', 'feature_charisma20', 'feature_charisma21', 'feature_charisma22', 'feature_charisma23', 'feature_charisma24', 'feature_charisma25', 'feature_charisma26', 'feature_charisma27', 'feature_charisma28', 'feature_charisma29', 'feature_charisma30', 'feature_charisma31', 'feature_charisma32', 'feature_charisma33', 'feature_charisma34', 'feature_charisma35', 'feature_charisma36', 'feature_charisma37', 'feature_charisma38', 'feature_charisma39', 'feature_charisma40', 'feature_charisma41', 'feature_charisma42', 'feature_charisma43', 'feature_charisma44', 'feature_charisma45', 'feature_charisma46', 'feature_charisma47', 'feature_charisma48', 'feature_charisma49', 'feature_charisma50', 'feature_charisma51', 'feature_charisma52', 'feature_charisma53', 'feature_charisma54', 'feature_charisma55', 'feature_charisma56', 'feature_charisma57', 'feature_charisma58', 'feature_charisma59', 'feature_charisma60', 'feature_charisma61', 'feature_charisma62', 'feature_charisma63', 'feature_charisma64', 'feature_charisma65', 'feature_charisma66', 'feature_charisma67', 'feature_charisma68', 'feature_charisma69', 'feature_charisma70', 'feature_charisma71', 'feature_charisma72', 'feature_charisma73', 'feature_charisma74', 'feature_charisma75', 'feature_charisma76', 'feature_charisma77', 'feature_charisma78', 'feature_charisma79', 'feature_charisma80', 'feature_charisma81', 'feature_charisma82', 'feature_charisma83', 'feature_charisma84', 'feature_charisma85', 'feature_charisma86', 'feature_strength1', 'feature_strength2', 'feature_strength3', 'feature_strength4', 'feature_strength5', 'feature_strength6', 'feature_strength7', 'feature_strength8', 'feature_strength9', 'feature_strength10', 'feature_strength11', 'feature_strength12', 'feature_strength13', 'feature_strength14', 'feature_strength15', 'feature_strength16', 'feature_strength17', 'feature_strength18', 'feature_strength19', 'feature_strength20', 'feature_strength21', 'feature_strength22', 'feature_strength23', 'feature_strength24', 'feature_strength25', 'feature_strength26', 'feature_strength27', 'feature_strength28', 'feature_strength29', 'feature_strength30', 'feature_strength31', 'feature_strength32', 'feature_strength33', 'feature_strength34', 'feature_strength35', 'feature_strength36', 'feature_strength37', 'feature_strength38', 'feature_dexterity1', 'feature_dexterity2', 'feature_dexterity3', 'feature_dexterity4', 'feature_dexterity5', 'feature_dexterity6', 'feature_dexterity7', 'feature_dexterity8', 'feature_dexterity9', 'feature_dexterity10', 'feature_dexterity11', 'feature_dexterity12', 'feature_dexterity13', 'feature_dexterity14', 'feature_constitution1', 'feature_constitution2', 'feature_constitution3', 'feature_constitution4', 'feature_constitution5', 'feature_constitution6', 'feature_constitution7', 'feature_constitution8', 'feature_constitution9', 'feature_constitution10', 'feature_constitution11', 'feature_constitution12', 'feature_constitution13', 'feature_constitution14', 'feature_constitution15', 'feature_constitution16', 'feature_constitution17', 'feature_constitution18', 'feature_constitution19', 'feature_constitution20', 'feature_constitution21', 'feature_constitution22', 'feature_constitution23', 'feature_constitution24', 'feature_constitution25', 'feature_constitution26', 'feature_constitution27', 'feature_constitution28', 'feature_constitution29', 'feature_constitution30', 'feature_constitution31', 'feature_constitution32', 'feature_constitution33', 'feature_constitution34', 'feature_constitution35', 'feature_constitution36', 'feature_constitution37', 'feature_constitution38', 'feature_constitution39', 'feature_constitution40', 'feature_constitution41', 'feature_constitution42', 'feature_constitution43', 'feature_constitution44', 'feature_constitution45', 'feature_constitution46', 'feature_constitution47', 'feature_constitution48', 'feature_constitution49', 'feature_constitution50', 'feature_constitution51', 'feature_constitution52', 'feature_constitution53', 'feature_constitution54', 'feature_constitution55', 'feature_constitution56', 'feature_constitution57', 'feature_constitution58', 'feature_constitution59', 'feature_constitution60', 'feature_constitution61', 'feature_constitution62', 'feature_constitution63', 'feature_constitution64', 'feature_constitution65', 'feature_constitution66', 'feature_constitution67', 'feature_constitution68', 'feature_constitution69', 'feature_constitution70', 'feature_constitution71', 'feature_constitution72', 'feature_constitution73', 'feature_constitution74', 'feature_constitution75', 'feature_constitution76', 'feature_constitution77', 'feature_constitution78', 'feature_constitution79', 'feature_constitution80', 'feature_constitution81', 'feature_constitution82', 'feature_constitution83', 'feature_constitution84', 'feature_constitution85', 'feature_constitution86', 'feature_constitution87', 'feature_constitution88', 'feature_constitution89', 'feature_constitution90', 'feature_constitution91', 'feature_constitution92', 'feature_constitution93', 'feature_constitution94', 'feature_constitution95', 'feature_constitution96', 'feature_constitution97', 'feature_constitution98', 'feature_constitution99', 'feature_constitution100', 'feature_constitution101', 'feature_constitution102', 'feature_constitution103', 'feature_constitution104', 'feature_constitution105', 'feature_constitution106', 'feature_constitution107', 'feature_constitution108', 'feature_constitution109', 'feature_constitution110', 'feature_constitution111', 'feature_constitution112', 'feature_constitution113', 'feature_constitution114', 'feature_wisdom1', 'feature_wisdom2', 'feature_wisdom3', 'feature_wisdom4', 'feature_wisdom5', 'feature_wisdom6', 'feature_wisdom7', 'feature_wisdom8', 'feature_wisdom9', 'feature_wisdom10', 'feature_wisdom11', 'feature_wisdom12', 'feature_wisdom13', 'feature_wisdom14', 'feature_wisdom15', 'feature_wisdom16', 'feature_wisdom17', 'feature_wisdom18', 'feature_wisdom19', 'feature_wisdom20', 'feature_wisdom21', 'feature_wisdom22', 'feature_wisdom23', 'feature_wisdom24', 'feature_wisdom25', 'feature_wisdom26', 'feature_wisdom27', 'feature_wisdom28', 'feature_wisdom29', 'feature_wisdom30', 'feature_wisdom31', 'feature_wisdom32', 'feature_wisdom33', 'feature_wisdom34', 'feature_wisdom35', 'feature_wisdom36', 'feature_wisdom37', 'feature_wisdom38', 'feature_wisdom39', 'feature_wisdom40', 'feature_wisdom41', 'feature_wisdom42', 'feature_wisdom43', 'feature_wisdom44', 'feature_wisdom45', 'feature_wisdom46', 'target']

if __name__ == '__main__':

    logs.configure_logging('info')
    logger = logging.getLogger(__name__)

    try:

        parser = argparse.ArgumentParser(description='Prediction submission script')
        parser.add_argument('--model-name', default=environ.get('MODEL_NAME'), required=False)
        parser.add_argument('--model-id', default=environ.get('MODEL_ID'), required=False)
        parser.add_argument('--neutralize', default=environ.get('NEUTRALIZE'), required=False)
        args = parser.parse_args()

        model_name = args.model_name
        model_id = args.model_id
        neutralize = args.neutralize
        models_path = abspath('outputs/models')
        submissions_path = abspath('outputs/submissions')

        logger.info('Creating directories')

        if not exists(submissions_path):
            logger.info(f'Creating submissions directory: {submissions_path}')
            makedirs(submissions_path)

        if not exists(models_path):
            logger.info(f'Creating models directory: {models_path}')
            makedirs(models_path)

        logger.info('Downloading tournament dataset')
        tournament_data = util.read_csv(file_path=dataset, column_names=column_names)
        feature_names = [f for f in tournament_data.columns if f.startswith('feature')]

        logger.info(f'Preparing model {model_name}')
        model = util.download_model(f'nomi_{model_name}.pkl', models_path)

        logger.info('Predicting')
        tournament_data.loc[:, 'prediction'] = model.predict(tournament_data[feature_names])

        if neutralize is not None:
            logger.info('Neutralizing predictions')

            tournament_data.loc[:, 'prediction'] = util.neutralize(
                df=tournament_data, by=feature_names, proportion=float(neutralize))

        util.save_predictions(
            df=tournament_data['prediction'],
            path=join(submissions_path, f'{model_name}_submission.csv')
        )

        if model_name is not None:
            numerai.submit_prediction(
                model_id=model_id,
                prediction_path=join(submissions_path, f'{model_name}_submission.csv')
            )

        logger.info('Getting metrics')
        util.get_metrics_table(
            df=tournament_data[tournament_data.data_type == 'validation'],
            table_title='Validation Scores'
        )

    except Exception as e:
        raise e
