import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FixedSizeList as List } from 'react-window';

import Button from 'components/kit/Button';
import Input from 'components/kit/Input';
import { isEqual } from 'shared/utils/data';
import { camelCaseToSentence } from 'shared/utils/string';

import DraggableListItem from './DraggableListItem';
import css from './Transfer.module.scss';

interface Props {
  defaultTargetEntries: string[];
  entries: string[];
  initialTargetEntries?: string[];
  onChange?: (targetList: string[]) => void;
  reorder?: boolean;
  sourceListTitle?: string;
  targetListTitle?: string;
}

const Transfer: React.FC<Props> = ({
  entries,
  defaultTargetEntries,
  initialTargetEntries,
  sourceListTitle = 'Source',
  targetListTitle = 'Target',
  reorder = true,
  onChange,
}: Props) => {
  const [targetEntries, setTargetEntries] = useState<string[]>(
    initialTargetEntries ?? defaultTargetEntries ?? [],
  );
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const hiddenEntries = useMemo(() => {
    const targetEntriesSet = new Set(targetEntries);
    return entries.filter((entry) => !targetEntriesSet.has(entry));
  }, [entries, targetEntries]);

  const filteredHiddenEntries = useMemo(() => {
    const regex = RegExp(searchTerm, 'i');
    return hiddenEntries.filter((entry) => regex.test(camelCaseToSentence(entry)));
  }, [hiddenEntries, searchTerm]);

  const filteredVisibleEntries = useMemo(() => {
    const regex = RegExp(searchTerm, 'i');
    return targetEntries.filter((entry) => regex.test(camelCaseToSentence(entry)));
  }, [targetEntries, searchTerm]);

  const moveToLeft = useCallback((transfer: string | string[]) => {
    if (Array.isArray(transfer)) {
      const transferSet = new Set(transfer);
      setTargetEntries((prev) => prev.filter((entry) => !transferSet.has(entry)));
    } else {
      setTargetEntries((prev) => prev.filter((entry) => transfer !== entry));
    }
  }, []);

  const moveToRight = useCallback((transfer: string | string[]) => {
    if (Array.isArray(transfer)) {
      setTargetEntries((prev) => [...prev, ...transfer]);
    } else {
      setTargetEntries((prev) => [...prev, transfer]);
    }
  }, []);

  const resetEntries = useCallback(() => {
    setTargetEntries(defaultTargetEntries);
  }, [defaultTargetEntries]);

  useEffect(() => {
    onChange?.(targetEntries);
  }, [onChange, targetEntries]);

  const renderEntry = useCallback(
    (entryName: string) => {
      const renameEntry = (): string => {
        switch (entryName) {
          case 'id':
            return 'ID';
          case 'startTime':
            return 'Started';
          case 'searcherType':
            return 'Searcher';
          case 'forkedFrom':
            return 'Forked';
          case 'numTrials':
            return 'Trials';
          default:
            return camelCaseToSentence(entryName);
        }
      };
      const sentenceEntryName = renameEntry();
      const regex = new RegExp(searchTerm, 'i');
      if (searchTerm === '' || !regex.test(sentenceEntryName)) {
        return <span>{sentenceEntryName}</span>;
      }
      const searchIndex = sentenceEntryName.search(regex);
      return (
        <span>
          {sentenceEntryName.slice(0, searchIndex)}
          <mark>{sentenceEntryName.match(regex)?.[0]}</mark>
          {sentenceEntryName.slice(searchIndex + searchTerm.length)}
        </span>
      );
    },
    [searchTerm],
  );

  const renderRow = useCallback(
    (row: string, style: React.CSSProperties, handleClick: () => void) => {
      return (
        <li style={style} onClick={handleClick}>
          {renderEntry(row)}
        </li>
      );
    },
    [renderEntry],
  );

  const switchRowOrder = useCallback(
    (entry: string, newNeighborEntry: string) => {
      if (entry !== newNeighborEntry) {
        const updatedVisibleEntries = [...targetEntries];
        const entryIndex = updatedVisibleEntries.findIndex((entryName) => entryName === entry);
        const newNeighborEntryIndex = updatedVisibleEntries.findIndex(
          (entryName) => entryName === newNeighborEntry,
        );
        updatedVisibleEntries.splice(entryIndex, 1);
        updatedVisibleEntries.splice(newNeighborEntryIndex, 0, entry);
        setTargetEntries(updatedVisibleEntries);
      }
      return;
    },
    [targetEntries],
  );

  const renderDraggableRow = useCallback(
    (
      row: string,
      index: number,
      style: React.CSSProperties,
      handleClick: (event: React.MouseEvent<Element, MouseEvent>) => void,
      handleDrop: (column: string, newNeighborColumnName: string) => void,
    ) => {
      return (
        <DraggableListItem
          columnName={row}
          index={index}
          style={style}
          onClick={handleClick}
          onDrop={handleDrop}>
          {renderEntry(row)}
        </DraggableListItem>
      );
    },
    [renderEntry],
  );

  const renderHiddenRow = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const row = filteredHiddenEntries[index];
      return renderRow(row, style, () => moveToRight(row));
    },
    [filteredHiddenEntries, moveToRight, renderRow],
  );

  const renderVisibleRow = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const row = filteredVisibleEntries[index];
      return reorder
        ? renderDraggableRow(row, index, style, () => moveToLeft(row), switchRowOrder)
        : renderRow(row, style, () => moveToLeft(row));
    },
    [filteredVisibleEntries, moveToLeft, renderDraggableRow, renderRow, reorder, switchRowOrder],
  );

  return (
    <div className={css.base}>
      <Input placeholder="Search entries..." onChange={handleSearch} />
      <div className={css.entries}>
        <div className={css.column}>
          <h2>{sourceListTitle}</h2>
          <List
            className={css.listContainer}
            height={200}
            innerElementType="ul"
            itemCount={filteredHiddenEntries.length}
            itemSize={24}
            width="100%">
            {renderHiddenRow}
          </List>
          <Button type="link" onClick={() => moveToRight(filteredHiddenEntries)}>
            Add All
          </Button>
        </div>
        <div className={css.column}>
          <div className={css.targetTitleRow}>
            <h2>{targetListTitle}</h2>
            {!isEqual(defaultTargetEntries, targetEntries) && (
              <Button type="link" onClick={resetEntries}>
                Reset
              </Button>
            )}
          </div>
          <List
            className={css.listContainer}
            height={200}
            innerElementType="ul"
            itemCount={filteredVisibleEntries.length}
            itemSize={24}
            width="100%">
            {renderVisibleRow}
          </List>
          <Button
            type="link"
            onClick={() => {
              moveToLeft(filteredVisibleEntries);
              // removing everything was keeping the columns out of sync with the UI...
              moveToRight(['id', 'name']);
            }}>
            Remove All
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Transfer;
