import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Separator } from '../components/ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from '../components/ui/tooltip'
import {
  Brain,
  FileText,
  Globe,
  Lock,
  Shield,
  ArrowLeft,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

import { PoolOverview } from '../components/rag/PoolOverview'
import { ShareToPoolButton } from '../components/rag/ShareToPoolButton'
import { SyncStatusBadge, type SyncStatus } from '../components/rag/SyncStatusBadge'
import { PoolVisibilityToggle } from '../components/rag/PoolVisibilityToggle'

// ── Mock Data ───────────────────────────────────────────────────

interface MockDocument {
  id: string
  name: string
  chunkCount: number
  isShared: boolean
  syncStatus: SyncStatus
  chunks: MockChunk[]
}

interface MockChunk {
  id: string
  preview: string
  isPublic: boolean
}

const INITIAL_DOCUMENTS: MockDocument[] = [
  {
    id: 'doc1',
    name: 'Bittensor Whitepaper.pdf',
    chunkCount: 24,
    isShared: true,
    syncStatus: 'synced',
    chunks: [
      { id: 'c1a', preview: 'Bittensor is a decentralized machine learning network that creates an open marketplace for AI models...', isPublic: true },
      { id: 'c1b', preview: 'The TAO token serves as the native cryptocurrency, incentivizing miners who contribute useful intelligence...', isPublic: true },
      { id: 'c1c', preview: 'Yuma Consensus aggregates validator weights into final miner scores using a stake-weighted average...', isPublic: true },
      { id: 'c1d', preview: 'Subnets are specialized task networks within the Bittensor ecosystem, each focused on a specific domain...', isPublic: false },
    ],
  },
  {
    id: 'doc2',
    name: 'RAG Architecture Notes.md',
    chunkCount: 18,
    isShared: false,
    syncStatus: 'private',
    chunks: [
      { id: 'c2a', preview: 'Retrieval-Augmented Generation enhances LLM outputs by first retrieving relevant documents from a knowledge base...', isPublic: false },
      { id: 'c2b', preview: 'Vector databases like Qdrant store documents as high-dimensional embeddings for similarity-based retrieval...', isPublic: false },
      { id: 'c2c', preview: 'The main advantage of RAG over fine-tuning is that the knowledge base can be updated without retraining...', isPublic: false },
    ],
  },
  {
    id: 'doc3',
    name: 'Offline-First AI Systems.pdf',
    chunkCount: 31,
    isShared: true,
    syncStatus: 'syncing',
    chunks: [
      { id: 'c3a', preview: 'Offline-first AI systems process data locally without requiring constant internet connectivity...', isPublic: true },
      { id: 'c3b', preview: 'Local AI inference using Ollama allows users to run language models on consumer hardware...', isPublic: true },
      { id: 'c3c', preview: 'Offline-first architectures reduce latency by eliminating network round trips and API fees...', isPublic: false },
    ],
  },
  {
    id: 'doc4',
    name: 'Personal Research Journal.txt',
    chunkCount: 12,
    isShared: false,
    syncStatus: 'private',
    chunks: [
      { id: 'c4a', preview: 'Meeting notes from the decentralized AI workshop — key takeaways on incentive alignment...', isPublic: false },
      { id: 'c4b', preview: 'Ideas for improving knowledge deduplication using content-addressable hashing...', isPublic: false },
    ],
  },
]

// ── Main Component ──────────────────────────────────────────────

export default function PoolDemo() {
  const [documents, setDocuments] = useState<MockDocument[]>(INITIAL_DOCUMENTS)

  // Computed stats
  const totalChunks = documents.reduce((sum, d) => sum + d.chunkCount, 0)
  const publicChunks = documents.filter(d => d.isShared).reduce((sum, d) => sum + d.chunkCount, 0)
  const publicDocs = documents.filter(d => d.isShared).length
  const totalDocs = documents.length

  const handleShare = useCallback((docId: string) => {
    setDocuments(prev =>
      prev.map(d => {
        if (d.id !== docId) return d
        return {
          ...d,
          isShared: true,
          syncStatus: 'pending' as SyncStatus,
          chunks: d.chunks.map(c => ({ ...c, isPublic: true })),
        }
      })
    )
    // Simulate sync progression
    setTimeout(() => {
      setDocuments(prev =>
        prev.map(d => d.id === docId ? { ...d, syncStatus: 'syncing' as SyncStatus } : d)
      )
    }, 800)
    setTimeout(() => {
      setDocuments(prev =>
        prev.map(d => d.id === docId ? { ...d, syncStatus: 'synced' as SyncStatus } : d)
      )
    }, 2500)
  }, [])

  const handleMakePrivate = useCallback((docId: string) => {
    setDocuments(prev =>
      prev.map(d => {
        if (d.id !== docId) return d
        return {
          ...d,
          isShared: false,
          syncStatus: 'private' as SyncStatus,
          chunks: d.chunks.map(c => ({ ...c, isPublic: false })),
        }
      })
    )
  }, [])

  const handleToggleChunk = useCallback((docId: string, chunkId: string) => {
    setDocuments(prev =>
      prev.map(d => {
        if (d.id !== docId) return d
        return {
          ...d,
          chunks: d.chunks.map(c =>
            c.id === chunkId ? { ...c, isPublic: !c.isPublic } : c
          ),
        }
      })
    )
  }, [])

  const handleSyncNow = useCallback(() => {
    setDocuments(prev =>
      prev.map(d => {
        if (d.syncStatus === 'pending' || d.syncStatus === 'syncing') {
          return { ...d, syncStatus: 'synced' as SyncStatus }
        }
        return d
      })
    )
  }, [])

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                <Brain className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="text-lg font-semibold">SuperBrain Knowledge Pool</h1>
                <p className="text-xs text-muted-foreground">
                  Private by default. Public by choice.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1.5">
                <Sparkles className="h-3 w-3" />
                Demo Mode
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        {/* Pool Overview */}
        <PoolOverview
          localChunks={totalChunks - publicChunks}
          localDocs={totalDocs - publicDocs}
          publicChunks={publicChunks}
          publicDocs={publicDocs}
          peerCount={3}
          lastSync={new Date(Date.now() - 2 * 60 * 1000)}
          onSyncNow={handleSyncNow}
        />

        {/* Documents List */}
        <div>
          <h2 className="text-sm font-medium text-muted-foreground mb-3 uppercase tracking-wide">
            Your Documents
          </h2>
          <div className="space-y-3">
            {documents.map(doc => (
              <Card key={doc.id} className={cn(
                'transition-all duration-300',
                doc.isShared && 'ring-1 ring-emerald-500/20'
              )}>
                <CardContent className="p-4">
                  {/* Document header row */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        'flex h-9 w-9 items-center justify-center rounded-lg',
                        doc.isShared
                          ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                          : 'bg-muted text-muted-foreground'
                      )}>
                        <FileText className="h-4 w-4" />
                      </div>
                      <div>
                        <h3 className="font-medium text-sm">{doc.name}</h3>
                        <p className="text-xs text-muted-foreground">
                          {doc.chunkCount} chunks
                          {doc.isShared && (
                            <span className="text-emerald-600 dark:text-emerald-400">
                              {' '} — shared to network
                            </span>
                          )}
                        </p>
                      </div>
                    </div>

                    <ShareToPoolButton
                      documentName={doc.name}
                      chunkCount={doc.chunkCount}
                      isShared={doc.isShared}
                      onShare={() => handleShare(doc.id)}
                      onMakePrivate={() => handleMakePrivate(doc.id)}
                      syncStatus={doc.syncStatus}
                      peerCount={doc.syncStatus === 'synced' ? 3 : undefined}
                    />
                  </div>

                  {/* Chunk preview with toggles */}
                  <div className="space-y-1.5">
                    {doc.chunks.map((chunk, idx) => (
                      <div
                        key={chunk.id}
                        className={cn(
                          'flex items-start gap-2.5 rounded-md border p-2.5 transition-all duration-200',
                          chunk.isPublic
                            ? 'border-emerald-500/20 bg-emerald-500/5'
                            : 'border-transparent bg-muted/50'
                        )}
                      >
                        <PoolVisibilityToggle
                          isPublic={chunk.isPublic}
                          onToggle={() => handleToggleChunk(doc.id, chunk.id)}
                          size="sm"
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-muted-foreground mb-0.5">
                            Chunk {idx + 1}/{doc.chunks.length}
                          </p>
                          <p className="text-sm leading-relaxed truncate">
                            {chunk.preview}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Privacy footer */}
        <div className="text-center py-6">
          <div className="inline-flex items-center gap-2 text-xs text-muted-foreground">
            <Shield className="h-3.5 w-3.5" />
            <span>
              All data stays on your device unless you explicitly choose to share.
              No tracking. No telemetry. No cloud.
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
