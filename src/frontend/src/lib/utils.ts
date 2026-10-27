import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number | undefined | null): string {
  if (num === undefined || num === null || isNaN(num)) return '0';
  return new Intl.NumberFormat('en-US').format(num);
}

export function formatPercent(value: number | undefined | null, isAlreadyPercentage: boolean = false): string {
  if (value === undefined || value === null || isNaN(value)) return '0.00%';
  const numericVal = isAlreadyPercentage ? value : value * 100;
  return `${numericVal.toFixed(2)}%`;
}
