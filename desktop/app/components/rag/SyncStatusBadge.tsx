import { Badge } from '../ui/badge'
import { Lock, Clock, RefreshCw, Globe, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export type SyncStatus = 'private' | 'pending' | 'syncing' | 'synced'

interface SyncStatusBadgeProps {
  status: SyncStatus
  peerCount?: number
  className?: string
}

const statusConfig: Record<SyncStatus, {
  label: string
  icon: typeof Lock
  badgeClass: string
  iconClass: string
}> = {
  private: {
    label: 'Private',
    icon: Lock,
    badgeClass: 'bg-muted text-muted-foreground border-muted',
    iconClass: '',
  },
  pending: {
    label: 'Pending Sync',
    icon: Clock,
    badgeClass: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20 dark:text-yellow-400',
    iconClass: '',
  },
  syncing: {
    label: 'Syncing',
    icon: RefreshCw,
    badgeClass: 'bg-blue-500/10 text-blue-600 border-blue-500/20 dark:text-blue-400',
    iconClass: 'animate-spin',
  },
  synced: {
    label: 'Synced',
    icon: Globe,
    badgeClass: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400',
    iconClass: '',
  },
}

export function SyncStatusBadge({ status, peerCount, className }: SyncStatusBadgeProps) {
  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <Badge
      variant="outline"
      className={cn(
        'gap-1.5 font-medium transition-all duration-300',
        config.badgeClass,
        className
      )}
    >
      <Icon className={cn('h-3 w-3', config.iconClass)} />
      <span>{config.label}</span>
      {status === 'synced' && (
        <>
          <Check className="h-3 w-3" />
          {peerCount !== undefined && peerCount > 0 && (
            <span className="text-xs opacity-75">({peerCount} peers)</span>
          )}
        </>
      )}
    </Badge>
  )
}
