"""Compact, accessible status icons and sample-first annotation rows."""
from functools import lru_cache
from pathlib import Path
from PySide6 import QtCore,QtGui


@lru_cache(maxsize=24)
def status_icon(roi_status,label_status):
    # Two 16px symbols: left = specimen ROI, right = plaque labels.
    pix=QtGui.QPixmap(36,18);pix.fill(QtCore.Qt.GlobalColor.transparent)
    p=QtGui.QPainter(pix);p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    def pen(color,width=1.6):p.setPen(QtGui.QPen(QtGui.QColor(color),width,QtCore.Qt.PenStyle.SolidLine,QtCore.Qt.PenCapStyle.RoundCap,QtCore.Qt.PenJoinStyle.RoundJoin))
    pen('#6ed6bc' if roi_status=='approved' else ('#e8ba65' if roi_status=='flagged' else '#9daec4'))
    points=[(8,1),(14,5),(14,12),(8,16),(2,12),(2,5)]
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x,y) for x,y in points]))
    if roi_status=='approved':p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(4.5,8.5),QtCore.QPointF(7,11),QtCore.QPointF(11.5,6)]))
    elif roi_status=='flagged':p.drawLine(8,5,8,9);p.drawPoint(8,12)
    else:p.drawEllipse(QtCore.QPointF(8,8.5),1.2,1.2)
    if label_status=='done':
        pen('#6ed6bc');p.drawEllipse(QtCore.QRectF(20,2,14,14))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(23,9),QtCore.QPointF(26,12),QtCore.QPointF(31,6)]))
    elif label_status=='draft':
        pen('#e8ba65',2.2);p.drawLine(QtCore.QPointF(23,13),QtCore.QPointF(31,5));p.drawLine(QtCore.QPointF(22,15),QtCore.QPointF(25,14))
    elif label_status in ('error','geometry_changed'):
        pen('#f0808a' if label_status=='error' else '#e8ba65');p.drawEllipse(QtCore.QRectF(20,2,14,14));p.drawLine(27,5,27,10);p.drawPoint(27,13)
    else:
        pen('#9daec4');p.drawEllipse(QtCore.QRectF(20,2,14,14))
    p.end();return QtGui.QIcon(pix)


def update_item(item,path,root,roi_status,label_status):
    relative=Path(path).relative_to(root)
    sample_first=relative.name in ('roi.ply','roi.vtp','geometry.npz') and len(relative.parts)>1
    if sample_first:
        sample=relative.parent.name
        context=list(relative.parent.parent.parts)
        if context and context[0].lower()=='samples':context.pop(0)
        text=sample+'\n'+' / '.join(context+[relative.name])
    else:text=str(relative)
    label_text={'new':'not started','draft':'draft','done':'approved','geometry_changed':'ROI changed — labels need review','error':'error loading labels'}.get(label_status,label_status)
    description=f'ROI: {roi_status}\nLabels: {label_text}\n\n{relative}\n{path}'
    item.setText(text);item.setIcon(status_icon(roi_status,label_status));item.setToolTip(description)
    item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole,description)
    item.setStatusTip(f'ROI: {roi_status} · Labels: {label_text} · {relative}')
