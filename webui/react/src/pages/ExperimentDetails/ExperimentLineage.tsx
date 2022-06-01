import React, { useCallback, useEffect, useState } from 'react';
import Tree from 'react-d3-tree';
import {
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

const ExperimentLineage: React.FC<Props> = ({ experiment }: Props) => {
  const [ lineage, setLineage ] = useState<RawNodeDatum>();
  const [ selectedExperiment, setSelectedExperiment ] = useState<ExperimentBase>(experiment);
  const [ previousExperiment, setPreviousExperiment ] = useState<ExperimentBase | null>(null);

  const fetchLineage = async (id: number) => {
    return {
      children: [ {
        children: [
          {
            children: [
              {
                children: [],
                name: '3',
              },
            ],
            name: '2',
          },
          {
            children: [
              {
                children: [],
                name: '5',
              },
            ],
            name: '4',
          },
        ],
        name: '1',
      } ],
      name: id.toString(),
    };
  };

  const getLineage = useCallback(async (id: number) => {
    const lineageRes = await fetchLineage(id);
    setLineage(lineageRes);
  }, [ ]);

  useEffect(() => {
    getLineage(experiment.id);
  }, [ getLineage, experiment.id ]);

  const fetchPreviousExperiment = useCallback(async (id) => {
    if (!id) return;
    const previousExp = await getExperimentDetails({ id });
    setPreviousExperiment(previousExp);
  }, []);

  const nodeClickHandler = useCallback(async ({ name }) => {
    const selectedExp = await getExperimentDetails({ id: name });
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
    const fillColor = parseInt(nodeDatum.name) === selectedExperiment.id ? '#3E6FED' : 'black';
    return (
      <g>
        <circle fill={fillColor} r="15" onClick={() => nodeClickHandler(nodeDatum)} />
        <text fill="black" strokeWidth="1" x="20" onClick={toggleNode}>
          {nodeDatum.name}
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
        #{selectedExperiment.id}<br />
        {selectedExperiment.name}<br />
        Hyperparameters<br /><br />
        {JSON.stringify(selectedExperiment.hyperparameters)}
      </div>
    </div>
  );
};

export default ExperimentLineage;
