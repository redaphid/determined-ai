import { Table } from 'antd';
import React, { useCallback, useEffect, useState } from 'react';
import Tree from 'react-d3-tree';
import {
  experimentLineage,
  getExperimentDetails,
} from 'services/api';
import { ExperimentBase, TrialDetails } from 'types';

import css from './ExperimentLineage.module.scss';

export interface Props {
  experiment: ExperimentBase;
  trial?: TrialDetails;
}

interface RawNodeDatum {
  attributes?: Record<string, string | number | boolean>;
  children?: RawNodeDatum[];
  name: string;
}

interface TableProps {
  previousExperiment: ExperimentBase | null;
  selectedExperiment: ExperimentBase;
}

const AboutTable: React.FC<TableProps> =
({ selectedExperiment, previousExperiment }: TableProps) => {
  const columns = [
    { dataIndex: 'field', title: 'About', width: 200 },
    {
      className: css.selectedColumn,
      dataIndex: 'selectedValue',
      title: `#${selectedExperiment.id}`,
      width: 250,
    },
  ];
  if (previousExperiment) {
    columns.push({
      dataIndex: 'previousValue',
      title: `#${previousExperiment.id}`,
      width: 250,
    });
  }

  const dataSource = [ {
    field: 'startTime',
    previousValue: selectedExperiment.startTime,
    selectedValue: selectedExperiment.startTime,
  },
  {
    field: 'endTime',
    previousValue: selectedExperiment.endTime,
    selectedValue: selectedExperiment.endTime,
  } ];

  return <Table columns={columns} dataSource={dataSource} pagination={false} rowKey="field" />;
};

const HyperparametersTable: React.FC<TableProps> =
({ selectedExperiment, previousExperiment }: TableProps) => {
  let hyperparameterKeys = Object.keys(selectedExperiment.hyperparameters);
  if (previousExperiment) {
    const prevKeys = Object.keys(previousExperiment.hyperparameters);
    hyperparameterKeys = hyperparameterKeys.concat(prevKeys);
  }
  hyperparameterKeys = Array.from(new Set(hyperparameterKeys));

  const columns = [
    { dataIndex: 'parameterName', title: 'Hyperparameter', width: 200 },
    {
      className: css.selectedColumn,
      dataIndex: 'selectedParameter',
      title: `#${selectedExperiment.id}`,
      width: 250,
    },
  ];
  if (previousExperiment) {
    columns.push({
      dataIndex: 'previousParameter',
      title: `#${previousExperiment.id}`,
      width: 250,
    });
  }

  const dataSource = hyperparameterKeys.map(key => {
    return {
      parameterName: key,
      previousParameter: previousExperiment?.hyperparameters[key]?.val,
      selectedParameter: selectedExperiment?.hyperparameters[key]?.val,
    };
  });

  return (
    <Table columns={columns} dataSource={dataSource} pagination={false} rowKey="parameterName" />
  );
};

const ExperimentLineage: React.FC<Props> = ({ experiment }: Props) => {
  const [ lineage, setLineage ] = useState<RawNodeDatum>();
  const [ selectedExperiment, setSelectedExperiment ] = useState<ExperimentBase>(experiment);
  const [ previousExperiment, setPreviousExperiment ] = useState<ExperimentBase | null>(null);

  const getLineage = useCallback(async (id) => {
    const lineageRes = await experimentLineage({ experimentId: id });
    if (lineageRes) {
      setLineage(lineageRes.root);
    }
  }, [ ]);

  useEffect(() => {
    getLineage(experiment.id);
  }, [ getLineage, experiment.id ]);

  const fetchPreviousExperiment = useCallback(async (id) => {
    if (!id) return;
    const previousExp = await getExperimentDetails({ id });
    setPreviousExperiment(previousExp);
  }, []);

  const nodeClickHandler = useCallback(async ({ id }) => {
    const selectedExp = await getExperimentDetails({ id });
    setSelectedExperiment(selectedExp);
  }, [ ]);

  useEffect(() => {
    fetchPreviousExperiment(selectedExperiment.forkedFrom);
  }, [ fetchPreviousExperiment, selectedExperiment ]);

  const renderNodeWithCustomEvents = useCallback(({
    nodeDatum,
    toggleNode,
    nodeClickHandler,
  }) => {
    const fillColor = parseInt(nodeDatum.id) === selectedExperiment.id ? '#3E6FED' : 'black';
    return (
      <g>
        <circle fill={fillColor} r="15" onClick={() => nodeClickHandler(nodeDatum)} />
        <text fill="black" strokeWidth="1" x="20" onClick={toggleNode}>
          {nodeDatum.id}
        </text>
      </g>
    );
  }, [ selectedExperiment.id ]);

  return (
    <div className={css.base}>
      <div className={css.TreeWrapper}>
        {lineage ? (
          <Tree
            branchNodeClassName="node__branch"
            collapsible={false}
            data={lineage}
            leafNodeClassName="node__leaf"
            renderCustomNodeElement={(rd3tProps) =>
              renderNodeWithCustomEvents({ ...rd3tProps, nodeClickHandler })
            }
            rootNodeClassName="node__root"
            translate={{ x: 100, y: 300 }}
          />
        ) : null}
      </div>
      <div className={css.DetailsWrapper}>
        <div className={css.section}>
          #{selectedExperiment.id}<br />
          {selectedExperiment.name}<br />
        </div>
        <div className={css.section}>
          <AboutTable
            previousExperiment={previousExperiment}
            selectedExperiment={selectedExperiment}
          />
        </div>
        <div className={css.section}>
          <HyperparametersTable
            previousExperiment={previousExperiment}
            selectedExperiment={selectedExperiment}
          />
        </div>
      </div>
    </div>
  );
};

export default ExperimentLineage;
