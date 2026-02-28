import { Lock, Globe } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip'
import { cn } from '@/lib/utils'

interface PoolVisibilityToggleProps {
  isPublic: boolean
  onToggle: () => void
  size?: 'sm' | 'md'
  className?: string
}

export function PoolVisibilityToggle({
  isPublic,
  onToggle,
  size = 'sm',
  className,
}: PoolVisibilityToggleProps) {
  const iconSize = size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4'
  const buttonSize = size === 'sm' ? 'h-7 w-7' : 'h-8 w-8'

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={onToggle}
          className={cn(
            'inline-flex items-center justify-center rounded-md transition-all duration-200',
            buttonSize,
            isPublic
              ? 'bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/20 dark:text-emerald-400 ring-1 ring-emerald-500/20'
              : 'bg-muted text-muted-foreground hover:bg-muted/80 ring-1 ring-transparent hover:ring-muted-foreground/20',
            className
          )}
        >
          {isPublic ? (
            <Globe className={iconSize} />
          ) : (
            <Lock className={iconSize} />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="text-xs">
        {isPublic ? 'Public — click to make private' : 'Private — click to share'}
      </TooltipContent>
    </Tooltip>
  )
}
