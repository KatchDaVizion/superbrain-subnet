import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Separator } from '../ui/separator'
import {
  Brain,
  Globe,
  Lock,
  RefreshCw,
  Users,
  ArrowRight,
  Clock,
  FileText,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface PoolOverviewProps {
  localChunks: number
  localDocs: number
  publicChunks: number
  publicDocs: number
  peerCount: number
  lastSync: Date | null
  onSyncNow?: () => void
  className?: string
}

function StatBlock({
  icon: Icon,
  iconClass,
  label,
  count,
  sublabel,
}: {
  icon: typeof Brain
  iconClass: string
  label: string
  count: number
  sublabel: string
}) {
  return (
    <div className="flex-1 rounded-lg border bg-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className={cn('flex h-8 w-8 items-center justify-center rounded-full', iconClass)}>
          <Icon className="h-4 w-4" />
        </div>
        <span className="text-sm font-medium text-muted-foreground">{label}</span>
      </div>
      <div className="text-3xl font-bold tracking-tight">{count}</div>
      <div className="flex items-center gap-1.5 mt-1">
        <FileText className="h-3 w-3 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{sublabel}</span>
      </div>
    </div>
  )
}

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export function PoolOverview({
  localChunks,
  localDocs,
  publicChunks,
  publicDocs,
  peerCount,
  lastSync,
  onSyncNow,
  className,
}: PoolOverviewProps) {
  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Knowledge Pool</CardTitle>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Users className="h-3.5 w-3.5" />
            <span>{peerCount} peer{peerCount !== 1 ? 's' : ''} discovered</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Stats row */}
        <div className="flex gap-3">
          <StatBlock
            icon={Lock}
            iconClass="bg-muted text-muted-foreground"
            label="Local Knowledge"
            count={localChunks}
            sublabel={`${localDocs} document${localDocs !== 1 ? 's' : ''} — your eyes only`}
          />
          <div className="flex items-center">
            <ArrowRight className="h-4 w-4 text-muted-foreground" />
          </div>
          <StatBlock
            icon={Globe}
            iconClass="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
            label="Public Pool"
            count={publicChunks}
            sublabel={`${publicDocs} document${publicDocs !== 1 ? 's' : ''} — shared to network`}
          />
        </div>

        <Separator />

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <Clock className="h-3 w-3" />
              <span>
                {lastSync ? `Last sync: ${formatTimeAgo(lastSync)}` : 'Never synced'}
              </span>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onSyncNow}
            className="gap-1.5"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Sync Now
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
