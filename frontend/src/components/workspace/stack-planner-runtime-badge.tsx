"use client";

import { WaypointsIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { PRODUCT_ASSISTANT_ID, RUNTIME_NAME } from "@/core/branding";

import { Tooltip } from "./tooltip";

export function StackPlannerRuntimeBadge() {
  const accessibleLabel = `SP Runtime: assistant_id=${PRODUCT_ASSISTANT_ID} on ${RUNTIME_NAME}`;

  return (
    <Tooltip content={accessibleLabel}>
      <Badge
        aria-label={accessibleLabel}
        className="bg-background/80 h-6 border-dashed font-mono"
        data-assistant-id={PRODUCT_ASSISTANT_ID}
        data-testid="sp-runtime-indicator"
        tabIndex={0}
        variant="outline"
      >
        <WaypointsIcon />
        <span className="hidden sm:inline">SP Runtime</span>
        <span className="sm:hidden">SP</span>
      </Badge>
    </Tooltip>
  );
}
