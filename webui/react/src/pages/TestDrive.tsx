import { Empty } from 'antd';
import { Upload } from 'antd';
import { RcFile, UploadFile } from 'antd/lib/upload/interface';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router';

import HumanReadableNumber from 'components/HumanReadableNumber';
import ResponsiveTable from 'components/ResponsiveTable';
import Section from 'components/Section';
import { getExperimentDetails, getExpTrials } from 'services/api';
import imageIcon from 'shared/assets/images/icon-image.svg';
import Page from 'shared/components/Page';
import Spinner from 'shared/components/Spinner';
import { RecordKey } from 'shared/types';
import { isEqual } from 'shared/utils/data';
import { ExperimentBase, TrialDetails } from 'types';

import css from './TestDrive.module.scss';

interface Params {
  experimentId: string;
}

interface Prediction {
  label: string;
  [trial: number]: number;
}

const getUrl = (uuid: string) => `http://localhost:5000/checkpoint/${uuid}/test-drive`;

const sampleRenderer = () => {
  return <img src={imageIcon} />;
};

const predictionRenderer = (trialId: number) => (_: string, record: Prediction) => {
  return <HumanReadableNumber num={record[trialId]} />;
};

const TRIAL_LIMIT = 3;
const COLUMNS = [
  {
    dataIndex: 'label',
    key: 'label',
    sorter: (a: Prediction, b: Prediction): number => a.label.localeCompare(b.label),
    title: 'Label',
  },
  {
    dataIndex: 'sample',
    key: 'sample',
    render: sampleRenderer,
    title: 'Sample',
  },
];

const getBase64 = (file: RcFile): Promise<string> => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => resolve(reader.result as string);
  reader.onerror = error => reject(error);
});

const TestDrive: React.FC = () => {
  const { experimentId } = useParams<Params>();
  const [ canceler ] = useState(new AbortController());
  const [ fileList, setFileList ] = useState<UploadFile[]>([]);
  const [ experiment, setExperiment ] = useState<ExperimentBase>();
  const [ trials, setTrials ] = useState<TrialDetails[]>();
  const [ dataSource, setDataSource ] = useState<Prediction[]>([]);
  const [ isLoading, setIsLoading ] = useState(true);

  const checkpoints = useMemo(() => {
    return (trials || [])
      .reduce((acc, trial) => {
        const trialId = trial.id;
        const uuid = trial.bestAvailableCheckpoint?.uuid;
        return uuid ? [ ...acc, { trialId, uuid } ] : acc;
      }, [] as { trialId: number, uuid: string }[])
      .slice(0, TRIAL_LIMIT);
  }, [ trials ]);

  const columns = useMemo(() => {
    return [ ...COLUMNS, ...checkpoints.map(checkpoint => ({
      dataIndex: checkpoint.trialId,
      render: predictionRenderer(checkpoint.trialId),
      sorter: (a: Prediction, b: Prediction): number => {
        const predA = a[checkpoint.trialId];
        const predB = b[checkpoint.trialId];
        return predA - predB;
      },
      title: `Trial ${checkpoint.trialId} Prediction`,
    })) ];
  }, [ checkpoints ]);

  const fetchExperimentDetails = useCallback(async () => {
    try {
      const id = parseInt(experimentId);
      const options = { signal: canceler.signal };
      const experiment = await getExperimentDetails({ id }, options);
      setExperiment(prev => isEqual(experiment, prev) ? prev : experiment);
      console.log('experiment', experiment);

      const { trials } = await getExpTrials({
        id,
        orderBy: 'ORDER_BY_DESC',
        sortBy: 'SORT_BY_BEST_VALIDATION_METRIC',
        states: [ 'STATE_COMPLETED' ],
      }, options);
      console.log('trials', trials);
      if (trials.length !== 0) setTrials(trials);
    } catch (e) {
      console.error('fetch error', e);
    }
  }, [ experimentId, canceler.signal ]);

  const handleChange = useCallback(({ fileList: newFileList }) => {
    setFileList(newFileList);
  }, []);

  const handlePreview = useCallback(async (file: UploadFile) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj as RcFile);
    }
  }, []);

  const handleCustomRequest = useCallback(async (options) => {
    setIsLoading(true);

    const { file, onError, onProgress, onSuccess } = options;

    try {
      const formData = new FormData();
      formData.append('image', file);

      const responses = await Promise.all(checkpoints.map(checkpoint => {
        return fetch(getUrl(checkpoint.uuid), {
          // automagically sets Content-Type: multipart/form-data header
          body: formData,
          method: 'POST',
        });
      }));

      const labelMap: Record<RecordKey, Record<number, number>> = {};
      for (const [ index, response ] of responses.entries()) {
        const trialId = checkpoints[index].trialId;
        const json = await response.json();
        const predictions: [ number, string ][] = json.pred;
        for (const [ prediction, label ] of predictions) {
          labelMap[label] = labelMap[label] ?? {};
          labelMap[label][trialId] = prediction;
        }
      }

      const dataSource = Object.keys(labelMap).reduce((acc, label) => {
        const prediction = { key: label, label, sample: 'x', ...labelMap[label] };
        return [ ...acc, prediction ];
      }, [] as Prediction[]);

      setDataSource(dataSource);
      setIsLoading(false);
      onProgress(100);
      onSuccess();
    } catch (e) {
      onError(e);
    }
  }, [ checkpoints ]);

  useEffect(() => {
    fetchExperimentDetails();

    return () => {
      canceler.abort();
    };
  }, [ canceler, fetchExperimentDetails ]);

  useEffect(() => {
    if (trials !== undefined) setIsLoading(false);
  }, [ trials ]);

  return (
    <Page title={`Test Drive for Experiment ${experimentId}`}>
      <Spinner spinning={isLoading}>
        <div className={css.base}>
          <Upload
            accept="image/*"
            className={css.upload}
            customRequest={handleCustomRequest}
            fileList={fileList}
            listType="picture-card"
            maxCount={1}
            name="image"
            onChange={handleChange}
            onPreview={handlePreview}>
            {Empty.PRESENTED_IMAGE_SIMPLE}
            <p className={css.title}>Click or drag file to this area to upload</p>
          </Upload>
          {dataSource.length !== 0 && (
            <Section title="Results">
              <ResponsiveTable columns={columns} dataSource={dataSource} />
            </Section>
          )}
        </div>
      </Spinner>
    </Page>
  );
};

export default TestDrive;
