import { Globe, Lock, Shield } from 'lucide-react'
import { Button } from '../ui/button'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '../ui/alert-dialog'
import { SyncStatusBadge, type SyncStatus } from './SyncStatusBadge'
import { cn } from '@/lib/utils'

interface ShareToPoolButtonProps {
  documentName: string
  chunkCount: number
  isShared: boolean
  onShare: () => void
  onMakePrivate: () => void
  syncStatus: SyncStatus
  peerCount?: number
  className?: string
}

export function ShareToPoolButton({
  documentName,
  chunkCount,
  isShared,
  onShare,
  onMakePrivate,
  syncStatus,
  peerCount,
  className,
}: ShareToPoolButtonProps) {
  if (isShared) {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        <SyncStatusBadge status={syncStatus} peerCount={peerCount} />
        <Button
          variant="ghost"
          size="sm"
          onClick={onMakePrivate}
          className="text-muted-foreground hover:text-foreground"
        >
          <Lock className="h-3.5 w-3.5 mr-1" />
          Make Private
        </Button>
      </div>
    )
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            'gap-1.5 hover:border-emerald-500/50 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors',
            className
          )}
        >
          <Globe className="h-3.5 w-3.5" />
          Share to Public Pool
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/10">
              <Shield className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <AlertDialogTitle>Share to SuperBrain Network?</AlertDialogTitle>
          </div>
          <AlertDialogDescription className="space-y-3">
            <p>
              This will make <strong>{chunkCount} chunks</strong> from{' '}
              <strong>{documentName}</strong> available to other SuperBrain nodes
              anonymously.
            </p>
            <p>
              Your data will be shared through encrypted sync channels. You can
              make it private again at any time.
            </p>
            <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
              <strong>Privacy note:</strong> Shared chunks contain only the text
              content — no personal information, file paths, or device
              identifiers are transmitted.
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onShare}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            <Globe className="h-4 w-4 mr-1.5" />
            Share {chunkCount} Chunks
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
