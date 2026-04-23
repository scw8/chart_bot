-- ============================================================
-- Stock Bot — Supabase 초기 설정 SQL
-- Supabase 대시보드 → SQL Editor에서 실행
-- ============================================================

-- 1. Storage 버킷 생성 (대시보드 Storage 탭에서 직접 생성 권장)
--    버킷 이름: ml-models
--    Public: false (비공개)
--    Allowed MIME types: application/octet-stream

-- 2. predictions 테이블 생성
CREATE TABLE IF NOT EXISTS public.predictions (
    id            uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker        text        NOT NULL,
    etf_name      text        NOT NULL,
    market        text        NOT NULL,         -- 'korea' | 'us'
    predicted_date date       NOT NULL,         -- 예측 대상 날짜 (내일)
    direction     text        NOT NULL,         -- '상승' | '하락' | '불확실'
    probability   numeric(6,4) NOT NULL,        -- 상승 확률 (예: 0.6230)
    actual_direction integer,                   -- 실제 결과: 1=상승, 0=하락, NULL=미확인
    is_correct    boolean,                      -- NULL → 결과 확인 후 업데이트
    created_at    timestamptz DEFAULT now()
);

-- 3. RLS 활성화
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- 4. RLS 정책 — service_role만 전체 접근 (GitHub Actions 전용)
CREATE POLICY "service_full_access"
ON public.predictions
TO service_role
USING (true)
WITH CHECK (true);

-- 5. 인덱스 (조회 성능)
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
    ON public.predictions (ticker, predicted_date DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_market
    ON public.predictions (market, predicted_date DESC);

-- ============================================================
-- Storage 버킷 RLS (스토리지는 대시보드에서 설정 권장)
-- 아래는 참고용 — Supabase SQL로 직접 설정하려면 실행
-- ============================================================

-- service_role만 ml-models 버킷에 접근 허용
INSERT INTO storage.buckets (id, name, public)
VALUES ('ml-models', 'ml-models', false)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "service_upload_models"
ON storage.objects FOR INSERT
TO service_role
WITH CHECK (bucket_id = 'ml-models');

CREATE POLICY "service_download_models"
ON storage.objects FOR SELECT
TO service_role
USING (bucket_id = 'ml-models');

CREATE POLICY "service_update_models"
ON storage.objects FOR UPDATE
TO service_role
USING (bucket_id = 'ml-models');

CREATE POLICY "service_delete_models"
ON storage.objects FOR DELETE
TO service_role
USING (bucket_id = 'ml-models');
