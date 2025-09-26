'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { DynamicTable, ColumnDef } from '@/components/DynamicTable';
import { CollectionData } from '@/lib/weaviate';
import { DeleteObjectsModal } from '@/components/DeleteObjectsModal';

interface CollectionViewProps {
  collectionName: string;
  properties: Array<{
    name: string;
    dataType: string[];
    description?: string;
  }>;
}

export function CollectionView({ collectionName, properties }: CollectionViewProps) {
  const [data, setData] = useState<CollectionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<{ property: string; order: 'asc' | 'desc' } | null>(null);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [offset, setOffset] = useState(0);
  const [canLoadMore, setCanLoadMore] = useState(true);
  const topRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const OBJECTS_PER_PAGE = 250;

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToTop = () => {
    topRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchData = useCallback(async (loadMore = false) => {
    if (!loadMore) {
      setLoading(true);
      setData([]);
      setOffset(0);
    } else {
      setLoadingMore(true);
    }

    try {
      const currentOffset = loadMore ? offset : 0;
      const url = new URL(`/api/collection/${collectionName}`, window.location.origin);
      if (sortConfig) {
        url.searchParams.set('sortProperty', sortConfig.property);
        url.searchParams.set('sortOrder', sortConfig.order);
      }
      url.searchParams.set('limit', String(OBJECTS_PER_PAGE));
      url.searchParams.set('offset', String(currentOffset));

      const response = await fetch(url);
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.details || result.error || 'Failed to fetch data');
      }

      const newData = result.data || [];
      
      setData(prevData => loadMore ? [...prevData, ...newData] : newData);
      setOffset(currentOffset + newData.length);
      setCanLoadMore(newData.length === OBJECTS_PER_PAGE);
      setError(null);
    } catch (err) {
      console.error('Error in CollectionView:', {
        collectionName,
        error: err instanceof Error ? {
          name: err.name,
          message: err.message,
          cause: err.cause,
          stack: err.stack
        } : err
      });
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [collectionName, sortConfig, offset]);

  useEffect(() => {
    fetchData(false);
  }, [sortConfig]);

  const handleLoadMore = () => {
    fetchData(true);
  };

  const handleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleDeleteClick = () => {
    if (selectionMode) {
      if (selectedIds.size > 0) {
        setDeleteModalOpen(true);
      } else {
        // Exit selection mode if no items are selected
        setSelectionMode(false);
      }
    } else {
      setSelectionMode(true);
    }
  };

  const handleDeleteConfirm = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/collection/${collectionName}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          objectIds: Array.from(selectedIds)
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.details || error.error || 'Failed to delete objects');
      }
      setDeleteModalOpen(false);
      setSelectionMode(false);
      setSelectedIds(new Set());
      await fetchData();
    } catch (err) {
      console.error('Error deleting objects:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete objects');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelSelection = () => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  };

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded" role="alert">
        <p>{error}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  const columns: ColumnDef[] = properties.map((prop) => ({
    key: prop.name,
    label: prop.name,
    dataType: prop.dataType,
    render: (value: unknown) => {
      if (Array.isArray(value)) {
        return value.join(', ');
      }
      return String(value);
    },
  }));

  const handleSort = (columnKey: string) => {
    const column = columns.find(col => col.key === columnKey);
    if (!column?.dataType?.includes('date')) return;

    setSortConfig(current => {
      if (current?.property === columnKey) {
        if (current.order === 'desc') {
          return { property: columnKey, order: 'asc' };
        }
        return null;
      }
      return { property: columnKey, order: 'desc' }; // First click sorts descending
    });
  };

  return (
    <div ref={topRef}>
      <div className="mb-4 flex justify-between">
        <div className="flex gap-2">
          <button
            onClick={handleDeleteClick}
            className={`px-4 py-2 rounded-md text-white font-medium ${
              selectionMode && selectedIds.size > 0
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-red-100 text-red-700 hover:bg-red-200'
            }`}
          >
            Delete
          </button>
          {selectionMode && (
            <button
              onClick={handleCancelSelection}
              className="px-4 py-2 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={scrollToBottom}
            className="px-4 py-2 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50"
          >
            Bottom
          </button>
        </div>
      </div>

      <DynamicTable 
        data={data} 
        columns={columns} 
        onSort={handleSort}
        sortConfig={sortConfig ? { 
          key: sortConfig.property, 
          direction: sortConfig.order 
        } : null}
        selectionMode={selectionMode}
        selectedIds={selectedIds}
        onSelect={handleSelect}
      />

      {canLoadMore && (
        <div ref={bottomRef} className="mt-4 flex justify-center items-center gap-4">
          <button
            onClick={handleLoadMore}
            disabled={loadingMore}
            className="px-6 py-2 rounded-md bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loadingMore ? 'Loading...' : 'Load More'}
          </button>
          <button
            onClick={scrollToTop}
            className="px-4 py-2 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50"
          >
            Top
          </button>
        </div>
      )}

      <DeleteObjectsModal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onDelete={handleDeleteConfirm}
        selectedCount={selectedIds.size}
      />
    </div>
  );
}
