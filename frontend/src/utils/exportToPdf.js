import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { calculateProfileFit, normalizeTop3Fits } from './profileFit';

/**
 * Generates and downloads a Career Path Recommendation PDF report.
 *
 * Layout (matches wireframe):
 *   ┌─────────────────────────────────────────┐
 *   │  CareerPath AI (logo)      Scan date    │  ← tinted header band
 *   ├─────────────────────────────────────────┤
 *   │  Career Path Recommendation Report      │
 *   │  ─────────────────────────────────────  │
 *   │  THE TOP 3 PREDICTED RESULT             │
 *   │    #1  #2  #3                           │
 *   │  ─────────────────────────────────────  │
 *   │  KEY INFLUENCING FACTORS                │
 *   │    keyword pills with strength bars     │
 *   │  ─────────────────────────────────────  │
 *   │  LEARNING ROADMAP                       │
 *   │  ─────────────────────────────────────  │
 *   │  CERTIFICATIONS                         │
 *   └─────────────────────────────────────────┘
 */
export const exportToPdf = async ({
    topThree,
    calculateProfileFit,
    learningRoadmap,
    certificationData,
    careerContent,
    logoRef,
    extractedKeywords,
    selectedCareerPath,
}) => {
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const PW = doc.internal.pageSize.getWidth();   // 210
    const PH = doc.internal.pageSize.getHeight();  // 297
    const ML = 20;
    const MR = 20;
    const CW = PW - ML - MR;
    const BOTTOM_MARGIN = 18;
    let y = 0;

    // ── helpers ───────────────────────────────────────────────────────────────

    /**
     * Replace Unicode characters that jsPDF's built-in Helvetica font cannot
     * encode, which causes it to silently fall back to Courier for the whole line.
     */
    const sanitize = (str) => String(str)
        .replace(/[\u2018\u2019\u201A\u201B\u2032\u2035]/g, "'")   // curly single quotes / prime
        .replace(/[\u201C\u201D\u201E\u201F\u2033\u2036]/g, '"')   // curly double quotes
        .replace(/[\u2013\u2014\u2015]/g, '-')                      // en-dash, em-dash, horizontal bar
        .replace(/\u2026/g, '...')                                  // ellipsis
        .replace(/[\u2022\u2023\u2043\u204C\u204D]/g, '-')         // bullets
        .replace(/[\u00A0\u202F\u2009\u200A]/g, ' ')               // non-breaking / thin spaces
        .replace(/[^\x00-\xFF]/g, '');                              // strip any remaining non-latin chars

    const checkPage = (need = 10) => {
        if (y + need > PH - BOTTOM_MARGIN) {
            doc.addPage();
            y = 20;
        }
    };

    /** Write wrapped text; returns total height consumed */
    const writeText = ({
        text,
        x = ML,
        maxW = CW,
        size = 9.5,
        style = 'normal',
        color = [50, 50, 50],
        lineH = 5,
    }) => {
        doc.setFont('helvetica', style);
        doc.setFontSize(size);
        doc.setTextColor(...color);
        const lines = doc.splitTextToSize(sanitize(text), maxW);
        lines.forEach(line => {
            checkPage(lineH + 2);
            // Re-apply font after checkPage in case a new page was added (jsPDF resets font on addPage)
            doc.setFont('helvetica', style);
            doc.setFontSize(size);
            doc.setTextColor(...color);
            doc.text(line, x, y);
            y += lineH;
        });
        return lines.length * lineH;
    };

    const thinRule = (color = [226, 232, 240]) => {
        doc.setDrawColor(...color);
        doc.setLineWidth(0.25);
        doc.line(ML, y, PW - MR, y);
    };

    // ── HEADER BAND ───────────────────────────────────────────────────────────
    doc.setFillColor(248, 250, 252);
    doc.rect(0, 0, PW, 22, 'F');

    // Logo — left  (capture the real rendered Logo component via html2canvas)
    if (logoRef?.current) {
        try {
            const canvas = await html2canvas(logoRef.current, {
                backgroundColor: null,
                scale: 3,
                logging: false,
            });
            const imgData = canvas.toDataURL('image/png');
            // Target height ~8mm in the band; compute width proportionally
            const logoH = 8;
            const logoW = (canvas.width / canvas.height) * logoH;
            doc.addImage(imgData, 'PNG', ML, 7, logoW, logoH);
        } catch {
            // Fallback: plain two-tone text
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(13);
            doc.setTextColor(30, 41, 59);
            doc.text('CareerPath', ML, 14);
            const cpW = doc.getTextWidth('CareerPath');
            doc.setTextColor(37, 99, 235);
            doc.text('-AI', ML + cpW, 14);
        }
    } else {
        // Fallback: plain two-tone text
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(30, 41, 59);
        doc.text('CareerPath', ML, 14);
        const cpW = doc.getTextWidth('CareerPath');
        doc.setTextColor(37, 99, 235);
        doc.text('-AI', ML + cpW, 14);
    }

    // Scan date — right
    const dateStr = new Date().toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric',
    });
    const reportYear = new Date().getFullYear();
    const dateLabel = `Scan Date: ${dateStr}`;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(120, 120, 120);
    doc.text(dateLabel, PW - MR - doc.getTextWidth(dateLabel), 14);

    // Bottom border of header band
    doc.setDrawColor(203, 213, 225);
    doc.setLineWidth(0.4);
    doc.line(0, 22, PW, 22);

    y = 33;

    // ── PAGE TITLE ────────────────────────────────────────────────────────────
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(17);
    doc.setTextColor(15, 23, 42);
    doc.text('Career Path Recommendation Report', ML, y);
    y += 12;

    // ── SECTION HEADER helper ─────────────────────────────────────────────────
    const sectionHeader = (label) => {
        checkPage(18);
        thinRule([203, 213, 225]);
        y += 8;
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(9.5);
        doc.setTextColor(71, 85, 105);
        doc.text(label, ML, y);
        y += 7;
    };

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 1 — TOP 3 PREDICTED RESULT
    // ══════════════════════════════════════════════════════════════════════════
    sectionHeader('THE TOP 3 PREDICTED RESULT');

    if (!topThree || topThree.length === 0) {
        writeText({
            text: 'No predicted paths available. Please complete the assessment to generate predictions.',
            style: 'normal',
            color: [120, 120, 120],
        });
    } else {
        const nFits = normalizeTop3Fits(topThree);
        topThree.forEach((career, idx) => {
            checkPage(28);

            const score = nFits[idx] ?? calculateProfileFit(career.raw_confidence);

            // Rank pill
            doc.setFillColor(239, 246, 255);
            doc.roundedRect(ML, y - 4.5, 7, 6.5, 1.2, 1.2, 'F');
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(9);
            doc.setTextColor(37, 99, 235);
            doc.text(`#${idx + 1}`, ML + 1.2, y);

            // Career name — wrapped so long titles never spill past the right margin
            const nameX = ML + 10;
            const nameMaxW = CW - 10;
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(11);
            doc.setTextColor(15, 23, 42);
            const nameLines = doc.splitTextToSize(sanitize(career.career_path), nameMaxW);
            nameLines.forEach(line => {
                doc.text(line, nameX, y);
                y += 6;
            });
            y += 1;

            // Progress bar track
            const barX = ML + 10;
            const scoreLabelW = 12;                        // reserved mm for "100%"
            const barW = CW - 10 - scoreLabelW - 3;       // guaranteed to fit
            const barH = 3.5;
            const fillW = (score / 100) * barW;

            // Track (background)
            doc.setFillColor(226, 232, 240);
            doc.roundedRect(barX, y - 2.5, barW, barH, 1, 1, 'F');

            // Fill — blue for #1, slightly lighter for #2/#3
            const fillColor = idx === 0 ? [37, 99, 235] : idx === 1 ? [59, 130, 246] : [147, 197, 253];
            doc.setFillColor(...fillColor);
            if (fillW > 0) {
                doc.roundedRect(barX, y - 2.5, fillW, barH, 1, 1, 'F');
            }

            // Score label right-aligned after bar
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(9);
            doc.setTextColor(...fillColor);
            doc.text(`${score}%`, barX + barW + 3, y + 0.5);
            y += 8;

            if (idx < topThree.length - 1) {
                y += 2;
            }
        });
    }

    y += 6;

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 2 — KEY INFLUENCING FACTORS
    // ══════════════════════════════════════════════════════════════════════════
    sectionHeader('KEY INFLUENCING FACTORS');

    if (selectedCareerPath) {
        writeText({
            text: `Factors for: ${sanitize(selectedCareerPath)}`,
            size: 9,
            color: [100, 116, 139],
            lineH: 4.8,
        });
        y += 2;
    }

    if (!extractedKeywords || extractedKeywords.length === 0) {
        writeText({
            text: 'No influencing keywords were extracted from your resume.',
            style: 'normal',
            color: [120, 120, 120],
        });
    } else {
        // Subtitle
        writeText({
            text: 'These keywords from your resume were most distinctive in predicting your top career path:',
            size: 9,
            color: [100, 116, 139],
            lineH: 4.8,
        });
        y += 4;

        const ITEM_H = 6.5;        // height of each row
        const COL_W = CW / 2 - 2; // two-column layout

        extractedKeywords.forEach((kw, ki) => {
            const col = ki % 2;           // 0 = left, 1 = right
            const xBase = ML + col * (COL_W + 4);

            // Start a new row every 2 items
            if (col === 0) checkPage(ITEM_H + 4);

            // Pill background
            doc.setFillColor(239, 246, 255);
            doc.roundedRect(xBase, y - 4.5, COL_W, ITEM_H, 1.5, 1.5, 'F');

            // Bullet dot
            doc.setFillColor(37, 99, 235);
            doc.circle(xBase + 4, y - 1.5, 1, 'F');

            // Keyword label
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(8.5);
            doc.setTextColor(30, 64, 175);
            const kwLabel = sanitize(kw.keyword);
            const kwLines = doc.splitTextToSize(kwLabel, COL_W - 10);
            doc.text(kwLines[0], xBase + 8, y - 0.5);

            // Advance y only after filling the right column (or at the last item)
            if (col === 1 || ki === extractedKeywords.length - 1) {
                y += ITEM_H + 2;
            }
        });
    }

    y += 6;

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 3 — LEARNING ROADMAP
    // ══════════════════════════════════════════════════════════════════════════
    sectionHeader('LEARNING ROADMAP');

    if (!learningRoadmap) {
        writeText({
            text: 'Learning roadmap has not been generated yet. Please open the Learning Roadmap panel in analysis page to generate it.',
            style: 'normal',
            color: [120, 120, 120],
        });
    } else {
        if (learningRoadmap.analysis_summary) {
            writeText({ text: sanitize(learningRoadmap.analysis_summary), size: 9, color: [100, 116, 139], lineH: 4.8 });
            y += 4;
        }

        (learningRoadmap.improvement_areas ?? []).forEach((area, aIdx) => {
            checkPage(20);

            doc.setFont('helvetica', 'bold');
            doc.setFontSize(10);
            doc.setTextColor(15, 23, 42);
            const skillLines = doc.splitTextToSize(sanitize(area.skill ?? 'Skill'), CW);
            skillLines.forEach(line => {
                checkPage(5.5);
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(10);
                doc.setTextColor(15, 23, 42);
                doc.text(line, ML, y);
                y += 5.5;
            });

            if (area.why) {
                writeText({ text: `Why: ${sanitize(area.why)}`, size: 9, color: [71, 85, 105], lineH: 4.6 });
                y += 2;
            }

            if (area.resources?.length > 0) {
                checkPage(10);
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(8.5);
                doc.setTextColor(100, 116, 139);
                doc.text('Resources', ML + 2, y);
                y += 5;

                area.resources.forEach(r => {
                    checkPage(6);
                    writeText({
                        text: `• ${sanitize(r.title)}  -  ${sanitize(r.provider)}  (${sanitize(r.type)})`,
                        x: ML + 4, maxW: CW - 4, size: 8.5, color: [71, 85, 105], lineH: 4.4,
                    });
                });
                y += 2;
            }

            if (aIdx < (learningRoadmap.improvement_areas.length - 1)) {
                thinRule([226, 232, 240]);
                y += 5;
            }
        });
    }

    y += 6;

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 4 — CERTIFICATIONS
    // ══════════════════════════════════════════════════════════════════════════
    sectionHeader('CERTIFICATIONS');

    if (!certificationData) {
        writeText({
            text: 'Certifications have not been generated yet. Please open the Certifications panel in analysis page to generate them.',
            style: 'normal',
            color: [120, 120, 120],
        });
    } else {
        if (certificationData.summary) {
            writeText({ text: sanitize(certificationData.summary), size: 9, color: [100, 116, 139], lineH: 4.8 });
            y += 4;
        }

        const levelColors = {
            beginner: [16, 185, 129],
            intermediate: [245, 158, 11],
            advanced: [239, 68, 68],
        };

        (certificationData.certifications ?? []).forEach((cert, cIdx) => {
            checkPage(24);

            // Name (left) — wrapped; level tag (right) on first line
            const certName = sanitize(cert.name ?? 'Certification');
            const lvl = cert.level?.toLowerCase() ?? 'intermediate';
            const [lr, lg, lb] = levelColors[lvl] ?? [100, 100, 100];
            const lvlTxt = sanitize(cert.level ?? '').toUpperCase();
            const lvlTxtW = doc.getTextWidth(lvlTxt) + 1;

            doc.setFont('helvetica', 'bold');
            doc.setFontSize(10.5);
            doc.setTextColor(15, 23, 42);
            const certNameLines = doc.splitTextToSize(certName, CW - lvlTxtW - 4);
            certNameLines.forEach((line, li) => {
                checkPage(5.5);
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(10.5);
                doc.setTextColor(15, 23, 42);
                doc.text(line, ML, y);
                if (li === 0) {
                    // level badge on same baseline as first line
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(8);
                    doc.setTextColor(lr, lg, lb);
                    doc.text(lvlTxt, PW - MR - doc.getTextWidth(lvlTxt), y);
                }
                y += 5.5;
            });

            const meta = [cert.provider, cert.estimated_duration].filter(Boolean).map(sanitize).join('  -  ');
            if (meta) {
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(8.5);
                doc.setTextColor(100, 116, 139);
                doc.text(meta, ML, y);
                y += 4.8;
            }

            if (cert.why) {
                writeText({ text: `Why: ${sanitize(cert.why)}`, size: 9, color: [71, 85, 105], lineH: 4.6 });
                y += 2;
            }

            if (cIdx < (certificationData.certifications.length - 1)) {
                thinRule([226, 232, 240]);
                y += 5;
            }
        });
    }

    // ── FOOTER ────────────────────────────────────────────────────────────────
    const totalPages = doc.internal.getNumberOfPages();
    for (let p = 1; p <= totalPages; p++) {
        doc.setPage(p);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.5);
        doc.setTextColor(148, 163, 184);
        doc.text(
            `CareerPath AI © ${reportYear}  •  Career Path Recommendation Report  •  Page ${p} of ${totalPages}`,
            PW / 2, PH - 8, { align: 'center' }
        );
    }

    // ── SAVE ──────────────────────────────────────────────────────────────────
    const today = new Date().toISOString().slice(0, 10);
    doc.save(`career-path-report_${today}.pdf`);
};
